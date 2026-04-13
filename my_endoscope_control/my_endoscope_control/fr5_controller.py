#!/usr/bin/env python3
"""
fr5_controller.py
入口 + 交互层：组合 FR5TrajectoryPlanner 和 FR5MotionController。
负责命令行交互，自身不包含规划或执行逻辑。

用法：
cd ~/ros2_ws && colcon build --packages-select my_endoscope_control
source install/setup.bash
ros2 run my_endoscope_control fr5_controller
  

命令：
  0~3        → 运动到预设位置
  rx <角度>  → 绕EE X轴旋转（保持末端位置不变）
  ry <角度>  → 绕EE Y轴旋转
  rz <角度>  → 绕EE Z轴旋转
  fk         → 打印当前EE位姿
  p          → 取消当前运动
  q          → 退出
"""

import sys
import time
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor

from my_endoscope_control.fr5_motion_controller  import FR5MotionController
from my_endoscope_control.fr5_trajectory_planner import FR5TrajectoryPlanner, PRESETS


# ─────────────────────────────────────────────
# 帮助文本
# ─────────────────────────────────────────────
def _print_help(presets: dict):
    print("\n" + "=" * 55)
    print("  FR5 内窥镜控制节点")
    print("=" * 55)
    print("  预设位置：")
    for k, name in presets.items():
        print(f"    {k}  →  {name}")
    print()
    print("  EE旋转（保持末端位置不变）：")
    print("    rx <角度>  →  绕EE X轴旋转（度）")
    print("    ry <角度>  →  绕EE Y轴旋转（度）")
    print("    rz <角度>  →  绕EE Z轴旋转（度）")
    print()
    print("  其他：")
    print("    fk         →  打印当前EE位姿")
    print("    p          →  取消当前运动")
    print("    h          →  显示本帮助")
    print("    q          →  退出")
    print("=" * 55)


# ─────────────────────────────────────────────
# 交互主循环（运行在独立线程，不阻塞 ROS spin）
# ─────────────────────────────────────────────
def input_loop(controller: FR5MotionController, planner: FR5TrajectoryPlanner):
    time.sleep(1.5)  # 等节点初始化完成
    _print_help(planner.list_presets())

    while rclpy.ok():
        try:
            raw = input("\n指令> ").strip()
        except EOFError:
            break

        if not raw:
            continue

        parts = raw.lower().split()
        cmd   = parts[0]

        # ── 退出 ──────────────────────────────
        if cmd == "q":
            print("退出...")
            rclpy.shutdown()
            break

        # ── 帮助 ──────────────────────────────
        elif cmd == "h":
            _print_help(planner.list_presets())

        # ── 取消运动 ──────────────────────────
        elif cmd == "p":
            controller.cancel_motion()

        # ── 正运动学打印 ──────────────────────
        elif cmd == "fk":
            joints = controller.get_current_joints()
            if joints is None:
                print("[错误] 尚未收到关节状态")
                continue
            j6 = [joints.get(j, 0.0) for j in ["j1","j2","j3","j4","j5","j6"]]
            fk = planner.forward_kinematics(j6)
            print(f"  当前关节角(rad): {[round(v,4) for v in j6]}")
            print(f"  EE 位置(m):      {fk['position']}")
            print(f"  EE 姿态(deg):    {fk['rpy_deg']}")

        # ── 预设位置 ──────────────────────────
        elif cmd in PRESETS:
            if controller.is_moving():
                print("[警告] 机械臂运动中，请先用 p 取消")
                continue

            target = planner.get_preset_joints(cmd)
            name   = planner.get_preset_name(cmd)
            print(f"[规划] → {name}  {[round(v,3) for v in target]}")
            controller.move_to_joints(target)

        # ── EE 旋转 ───────────────────────────
        elif cmd in ("rx", "ry", "rz"):
            if len(parts) < 2:
                print(f"用法：{cmd} <角度(度)>  例：rz 30")
                continue

            try:
                angle_deg = float(parts[1])
            except ValueError:
                print(f"[错误] 角度必须是数字，收到：{parts[1]!r}")
                continue

            if controller.is_moving():
                print("[警告] 机械臂运动中，请先用 p 取消")
                continue

            joints = controller.get_current_joints()
            if joints is None:
                print("[错误] 尚未收到关节状态")
                continue

            current6 = [joints.get(j, 0.0) for j in ["j1","j2","j3","j4","j5","j6"]]
            axis     = cmd[1]  # 'x' / 'y' / 'z'

            print(f"[规划] 绕EE {axis.upper()}轴旋转 {angle_deg:.1f}°（位置保持不变）")

            target = planner.plan_ee_rotation(current6, axis=axis, angle_deg=angle_deg)
            if target is None:
                print("[错误] IK 求解失败，可能超出工作空间或奇异点附近，请换个角度或先移动到其他位置")
                continue

            print(f"[规划] IK结果: {[round(v,4) for v in target]}")
            controller.move_to_joints(target)

        # ── 未知指令 ──────────────────────────
        else:
            print(f"未知指令：{raw!r}  输入 h 查看帮助")


# ─────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────
def main():
    rclpy.init()

    # 实例化两个模块
    controller = FR5MotionController()

    try:
        planner = FR5TrajectoryPlanner()
    except FileNotFoundError as e:
        print(f"[致命错误] {e}")
        controller.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # 输入循环放独立线程，不阻塞 ROS spin
    t = threading.Thread(
        target=input_loop,
        args=(controller, planner),
        daemon=True,
    )
    t.start()

    # 使用 MultiThreadedExecutor，让 Action 回调和订阅回调并发执行
    executor = MultiThreadedExecutor()
    executor.add_node(controller)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        controller.cleanup()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
