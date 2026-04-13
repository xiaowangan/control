#!/usr/bin/env python3
"""
fr5_motion_controller.py
纯执行层：负责接收关节目标，发送 Action，管理运动状态，记录CSV日志。
不包含任何轨迹规划逻辑。
"""

import csv
import math
import time
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────
JOINT_NAMES   = ["j1", "j2", "j3", "j4", "j5", "j6"]
MAX_JOINT_VEL = 0.8    # rad/s
MAX_JOINT_ACC = 0.4    # rad/s²
MIN_DURATION  = 1.0    # s
LOG_FILE      = "motion_log.csv"

ACTION_SERVER = "/fairino5_controller/follow_joint_trajectory"


class FR5MotionController(Node):
    """
    纯执行层节点。
    外部通过 move_to_joints(target, duration=None) 发送运动指令。
    通过 cancel_motion() 取消当前运动。
    通过 is_idle() / is_moving() 查询状态。
    通过 get_current_joints() 获取当前关节角度。
    """

    def __init__(self):
        super().__init__("fr5_motion_controller")

        # ── 关节状态 ──────────────────────────────
        self._joints_lock    = threading.Lock()
        self._current_joints = None          # dict {name: rad}

        # ── 运动状态 ──────────────────────────────
        self._state_lock         = threading.Lock()
        self._state              = "IDLE"    # IDLE / MOVING
        self._current_goal_handle = None

        # ── 订阅 /joint_states ────────────────────
        self.create_subscription(
            JointState, "/joint_states", self._joint_cb, 10
        )

        # ── Action client ─────────────────────────
        self._ac = ActionClient(self, FollowJointTrajectory, ACTION_SERVER)

        # ── CSV 日志 ──────────────────────────────
        self._csv_file = open(LOG_FILE, "w", newline="")
        self._csv      = csv.writer(self._csv_file)
        self._csv.writerow([
            "timestamp", "event",
            "j1_tgt","j2_tgt","j3_tgt","j4_tgt","j5_tgt","j6_tgt",
            "j1_act","j2_act","j3_act","j4_act","j5_act","j6_act",
            "j1_err","j2_err","j3_err","j4_err","j5_err","j6_err",
            "duration_s",
        ])

        self.get_logger().info("FR5MotionController 启动，等待 /joint_states ...")

    # ═══════════════════════════════════════════
    # 公开 API
    # ═══════════════════════════════════════════

    def get_current_joints(self) -> dict | None:
        """返回当前关节角度字典副本，尚未收到则返回 None。"""
        with self._joints_lock:
            return dict(self._current_joints) if self._current_joints else None

    def is_idle(self) -> bool:
        with self._state_lock:
            return self._state == "IDLE"

    def is_moving(self) -> bool:
        with self._state_lock:
            return self._state == "MOVING"

    def move_to_joints(self, target: list[float], duration: float | None = None) -> bool:
        """
        向目标关节角度运动。
        target   : 6个关节角度列表（弧度）
        duration : 运动时间(s)，None 则自动根据速度/加速度限制计算
        返回 True=指令已发送，False=发送失败
        """
        # ── 前置检查 ──────────────────────────────
        joints = self.get_current_joints()
        if joints is None:
            self.get_logger().error("尚未收到 /joint_states，无法运动")
            return False

        missing = [j for j in JOINT_NAMES if j not in joints]
        if missing:
            self.get_logger().error(f"关节状态缺失：{missing}")
            return False

        with self._state_lock:
            if self._state == "MOVING":
                self.get_logger().warn("机械臂正在运动，请先取消再发指令")
                return False

        current  = [joints[j] for j in JOINT_NAMES]
        duration = duration or self._calc_duration(current, target)

        # ── 等待 Action server ────────────────────
        if not self._ac.wait_for_server(timeout_sec=3.0):
            self.get_logger().error("Action server 未响应")
            return False

        # ── 构建轨迹消息 ──────────────────────────
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = JOINT_NAMES

        # 起始点（当前位置，t=0，速度=0）
        p0 = JointTrajectoryPoint()
        p0.positions     = current
        p0.velocities    = [0.0] * 6
        p0.accelerations = [0.0] * 6
        p0.time_from_start.sec     = 0
        p0.time_from_start.nanosec = 0

        # 终止点
        p1 = JointTrajectoryPoint()
        p1.positions     = target
        p1.velocities    = [0.0] * 6
        p1.accelerations = [0.0] * 6
        p1.time_from_start.sec     = int(duration)
        p1.time_from_start.nanosec = int((duration % 1) * 1e9)

        goal.trajectory.points = [p0, p1]

        # ── 发送 ──────────────────────────────────
        with self._state_lock:
            self._state = "MOVING"
            self._current_goal_handle = None

        t_start = time.time()
        future  = self._ac.send_goal_async(goal)
        future.add_done_callback(
            lambda f: self._on_goal_response(f, target, current, duration, t_start)
        )

        self.get_logger().info(
            f"[运动] 目标={[round(v,3) for v in target]}  时间={duration:.2f}s"
        )
        return True

    def cancel_motion(self):
        """取消当前正在执行的运动目标。"""
        with self._state_lock:
            gh = self._current_goal_handle
            moving = self._state == "MOVING"

        if not moving or gh is None:
            self.get_logger().info("当前无运动任务")
            return

        self.get_logger().info("[取消] 发送取消指令...")
        cancel_fut = gh.cancel_goal_async()
        cancel_fut.add_done_callback(self._on_cancel_done)

    # ═══════════════════════════════════════════
    # 内部回调
    # ═══════════════════════════════════════════

    def _joint_cb(self, msg: JointState):
        with self._joints_lock:
            self._current_joints = dict(zip(msg.name, msg.position))

    def _on_goal_response(self, future, target, current, duration, t_start):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("目标被控制器拒绝")
            with self._state_lock:
                self._state = "IDLE"
            return

        with self._state_lock:
            self._current_goal_handle = goal_handle

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._on_result(f, target, current, duration, t_start)
        )

    def _on_result(self, future, target, current, duration, t_start):
        elapsed = time.time() - t_start

        with self._state_lock:
            self._state = "IDLE"
            self._current_goal_handle = None

        status = future.result().status
        if status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("[取消] 运动已停止")
            return

        # 读取实际到达位置
        joints = self.get_current_joints()
        actual = [joints.get(j, 0.0) for j in JOINT_NAMES] if joints else target
        errors = [round(a - t, 5) for a, t in zip(actual, target)]
        max_err = max(abs(e) for e in errors)

        level = "OK" if max_err < 0.01 else "WARNING"
        self.get_logger().info(
            f"[完成] 耗时={elapsed:.2f}s  最大误差={max_err:.5f}rad  [{level}]"
        )
        self.get_logger().info(f"  误差(rad): {errors}")

        # 写 CSV
        self._csv.writerow([
            round(time.time(), 3), "MOVE_DONE",
            *[round(v, 5) for v in target],
            *[round(v, 5) for v in actual],
            *errors,
            round(elapsed, 3),
        ])
        self._csv_file.flush()

    def _on_cancel_done(self, future):
        code = future.result().return_code
        if code == 0:
            self.get_logger().info("[取消] 取消成功，机械臂停止")
        else:
            self.get_logger().warn(f"[取消] 取消被拒绝(code={code})，运动可能已完成")
        with self._state_lock:
            self._state = "IDLE"
            self._current_goal_handle = None

    # ═══════════════════════════════════════════
    # 工具函数
    # ═══════════════════════════════════════════

    @staticmethod
    def _calc_duration(current: list, target: list) -> float:
        t_min = MIN_DURATION
        for c, t in zip(current, target):
            delta = abs(t - c)
            t_min = max(t_min,
                        delta / MAX_JOINT_VEL,
                        math.sqrt(2.0 * delta / MAX_JOINT_ACC))
        return round(t_min, 2)

    def cleanup(self):
        self._csv_file.close()
        self.get_logger().info(f"[日志] 已保存到 {LOG_FILE}")
