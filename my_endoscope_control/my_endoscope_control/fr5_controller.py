#!/usr/bin/env python3
"""
FR5 运动控制节点
功能：预设位置运动 / 速度+加速度限制 / 误差终端打印 / 运动日志CSV
用法：python3 fr5_controller.py
      然后输入数字选择预设位置，输入 q 退出，输入 p 暂停/恢复
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
import csv
import time
import math
import threading

# ──────────────────────────────────────────────
# 预设位置（单位：弧度）
# 顺序对应 j1 j2 j3 j4 j5 j6
# ──────────────────────────────────────────────
PRESETS = {
    "0": {
        "name": "零位（home）",
        "joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    "1": {
        "name": "预设位置 1",
        "joints": [0.5, -0.5, 0.8, 0.0, 0.5, 0.0],
    },
    "2": {
        "name": "预设位置 2",
        "joints": [-0.5, -1.0, 1.0, 0.3, -0.5, 0.0],
    },
    "3": {
        "name": "预设位置 3",
        "joints": [1.0, -0.8, 0.6, -0.3, 0.8, 0.5],
    },
}

# ──────────────────────────────────────────────
# 运动参数
# ──────────────────────────────────────────────
MAX_JOINT_VEL   = 0.8   # rad/s  每个关节最大速度
MAX_JOINT_ACC   = 0.4   # rad/s² 每个关节最大加速度
MIN_DURATION    = 1.0   # s      最短运动时间
LOG_FILE        = "motion_log.csv"
JOINT_NAMES     = ["j1", "j2", "j3", "j4", "j5", "j6"]


class FR5Controller(Node):

    def __init__(self):
        super().__init__("fr5_controller")

        # 状态
        self.current_joints = None   # 当前关节角度 dict {name: value}
        self.state = "IDLE"          # IDLE / MOVING / PAUSED
        self.paused = False

        # 订阅关节状态
        self.sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10
        )

        # Action client
        self._ac = ActionClient(
            self, FollowJointTrajectory,
            "/fairino5_controller/follow_joint_trajectory"
        )

        # CSV 日志
        self._csv_file = open(LOG_FILE, "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "timestamp", "event",
            "j1_target", "j2_target", "j3_target",
            "j4_target", "j5_target", "j6_target",
            "j1_actual", "j2_actual", "j3_actual",
            "j4_actual", "j5_actual", "j6_actual",
            "j1_error",  "j2_error",  "j3_error",
            "j4_error",  "j5_error",  "j6_error",
            "duration_s"
        ])

        self.get_logger().info("FR5 控制节点启动，等待关节状态...")

    # ──────────────────────────────────────────
    # 回调：更新当前关节角度
    # ──────────────────────────────────────────
    def _joint_cb(self, msg: JointState):
        self.current_joints = dict(zip(msg.name, msg.position))

    # ──────────────────────────────────────────
    # 计算运动时间（基于速度/加速度限制）
    # ──────────────────────────────────────────
    def _calc_duration(self, current: list, target: list) -> float:
        t_min = MIN_DURATION
        for c, t in zip(current, target):
            delta = abs(t - c)
            # 速度限制
            t_vel = delta / MAX_JOINT_VEL
            # 加速度限制（三角形速度曲线近似）
            t_acc = math.sqrt(2.0 * delta / MAX_JOINT_ACC)
            t_min = max(t_min, t_vel, t_acc)
        return round(t_min, 2)

    # ──────────────────────────────────────────
    # 发送轨迹目标
    # ──────────────────────────────────────────
    def move_to(self, preset_key: str):
        if self.current_joints is None:
            print("[错误] 尚未收到关节状态，请稍候重试")
            return

        if self.state == "MOVING":
            print("[警告] 机械臂正在运动，请等待完成或暂停后再发指令")
            return

        preset = PRESETS[preset_key]
        target = preset["joints"]

        # 按名字取当前值（保证顺序正确）
        current = [self.current_joints.get(j, 0.0) for j in JOINT_NAMES]

        # 计算运动时间
        duration = self._calc_duration(current, target)

        print(f"\n[运动] → {preset['name']}")
        print(f"  目标: {[round(v,3) for v in target]}")
        print(f"  当前: {[round(v,3) for v in current]}")
        print(f"  预计时间: {duration:.2f}s  (速限:{MAX_JOINT_VEL}rad/s  加速限:{MAX_JOINT_ACC}rad/s²)")

        # 等待 Action server
        if not self._ac.wait_for_server(timeout_sec=3.0):
            print("[错误] Action server 未响应")
            return

        # 构建轨迹消息
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = target
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)
        goal.trajectory.points = [point]

        self.state = "MOVING"
        t_start = time.time()

        # 异步发送，用回调处理结果
        send_future = self._ac.send_goal_async(goal)
        send_future.add_done_callback(
            lambda f: self._goal_response_cb(f, target, current, duration, t_start)
        )

    # ──────────────────────────────────────────
    # Goal 接受回调
    # ──────────────────────────────────────────
    def _goal_response_cb(self, future, target, current, duration, t_start):
        goal_handle = future.result()
        if not goal_handle.accepted:
            print("[错误] 目标被拒绝")
            self.state = "IDLE"
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._result_cb(f, target, current, duration, t_start)
        )

    # ──────────────────────────────────────────
    # 运动完成回调
    # ──────────────────────────────────────────
    def _result_cb(self, future, target, current, duration, t_start):
        actual_duration = time.time() - t_start
        self.state = "IDLE"

        # 取执行后的实际位置
        actual = [self.current_joints.get(j, 0.0) for j in JOINT_NAMES]
        errors = [round(a - t, 5) for a, t in zip(actual, target)]

        # 终端打印误差
        print(f"\n[完成] 实际耗时: {actual_duration:.2f}s")
        print(f"  误差(rad): {errors}")
        max_err = max(abs(e) for e in errors)
        status = "OK" if max_err < 0.01 else "WARNING"
        print(f"  最大误差: {max_err:.5f} rad  [{status}]")

        # 写入CSV
        ts = round(time.time(), 3)
        self._csv.writerow([
            ts, "MOVE_DONE",
            *[round(v, 5) for v in target],
            *[round(v, 5) for v in actual],
            *errors,
            round(actual_duration, 3)
        ])
        self._csv_file.flush()

        print("\n输入数字选择下一个位置，p=暂停/恢复，q=退出")

    # ──────────────────────────────────────────
    # 清理
    # ──────────────────────────────────────────
    def cleanup(self):
        self._csv_file.close()
        print(f"\n[日志] 已保存到 {LOG_FILE}")


# ──────────────────────────────────────────────
# 主循环：命令行交互
# ──────────────────────────────────────────────
def input_loop(node: FR5Controller):
    """在单独线程里跑，不阻塞 ROS spin"""
    time.sleep(1.5)  # 等节点初始化

    print("\n" + "="*50)
    print("  FR5 运动控制节点")
    print("="*50)
    for k, v in PRESETS.items():
        print(f"  {k}  →  {v['name']}  {v['joints']}")
    print("  p  →  暂停 / 恢复")
    print("  q  →  退出")
    print("="*50)

    while rclpy.ok():
        try:
            cmd = input("\n指令> ").strip().lower()
        except EOFError:
            break

        if cmd == "q":
            print("退出...")
            rclpy.shutdown()
            break
        elif cmd == "p":
            node.paused = not node.paused
            node.state = "PAUSED" if node.paused else "IDLE"
            print(f"[状态] {'已暂停' if node.paused else '已恢复'}")
        elif cmd in PRESETS:
            node.move_to(cmd)
        else:
            print(f"未知指令：{cmd}，可用：{list(PRESETS.keys())} / p / q")


def main():
    rclpy.init()
    node = FR5Controller()

    # 输入循环放在独立线程
    t = threading.Thread(target=input_loop, args=(node,), daemon=True)
    t.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()


if __name__ == "__main__":
    main()