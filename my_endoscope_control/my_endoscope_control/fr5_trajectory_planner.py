#!/usr/bin/env python3
"""
fr5_trajectory_planner.py
纯规划层：负责生成关节角度目标。
包含：
  1. 预设位置查询
  2. 绕 EE 末端坐标系旋转的 IK 规划（自定义L-BFGS-B求解器，稳定可靠）
不依赖 ROS，可单独 python3 fr5_trajectory_planner.py 测试。
"""

import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import ikpy.chain

# ─────────────────────────────────────────────
# URDF 路径候选列表（按顺序查找第一个存在的）
# 如果都找不到，在列表末尾添加你的实际路径
# ─────────────────────────────────────────────
_URDF_CANDIDATES = [
    # 标准 ros2_ws 结构
    os.path.expanduser("~/ros2_ws/src/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf"),
    # ament install 后的路径
    os.path.expanduser("~/ros2_ws/install/fairino_description/share/fairino_description/urdf/fairino5_v6.urdf"),
    # 相对路径（从本文件往上找）
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "../../frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf"),
]


def _find_urdf() -> str:
    for p in _URDF_CANDIDATES:
        p = os.path.abspath(p)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "找不到 fairino5_v6.urdf！\n"
        "请在 fr5_trajectory_planner.py 顶部的 _URDF_CANDIDATES 列表中\n"
        "添加你的实际路径，例如：\n"
        "  '/home/an/ros2_ws/src/frcobot_ros2/fairino_description/urdf/fairino5_v6.urdf'"
    )


# ─────────────────────────────────────────────
# ikpy 链配置
# links: Base(F) j1~j6(T) endoscope_holder(F) endoscope_tip(F) last(F)
# ─────────────────────────────────────────────
_ACTIVE_MASK = [False, True, True, True, True, True, True, False, False, False]
JOINT_NAMES  = ["j1", "j2", "j3", "j4", "j5", "j6"]

# 关节限位（弧度），参考 FR5 规格书 ±175°
_JOINT_BOUNDS = [(-3.054, 3.054)] * 6

# ─────────────────────────────────────────────
# 预设位置（弧度）
# ─────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    "0": {"name": "零位（home）",  "joints": [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]},
    "1": {"name": "预设位置 1",    "joints": [0.5, -0.5,  0.8,  0.0,  0.5,  0.0]},
    "2": {"name": "预设位置 2",    "joints": [-0.5,-1.0,  1.0,  0.3, -0.5,  0.0]},
    "3": {"name": "预设位置 3",    "joints": [1.0, -0.8,  0.6, -0.3,  0.8,  0.5]},
}


class FR5TrajectoryPlanner:
    """
    纯规划类，无 ROS 依赖，可独立测试。

    主要接口：
        get_preset_joints(key)               → list[float] | None
        forward_kinematics(joints)           → dict
        plan_ee_rotation(joints, axis, deg)  → list[float] | None
        plan_ee_rotation_steps(...)          → list[list[float]] | None
    """

    def __init__(self):
        urdf_path = _find_urdf()
        self._chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            base_elements=["base_link"],
            last_link_vector=[0, 0, 0],
            active_links_mask=_ACTIVE_MASK,
        )
        print(f"[Planner] URDF 加载：{urdf_path}")

    # ═══════════════════════════════════════════
    # 公开 API
    # ═══════════════════════════════════════════

    def get_preset_joints(self, key: str) -> list[float] | None:
        p = PRESETS.get(key)
        return list(p["joints"]) if p else None

    def get_preset_name(self, key: str) -> str:
        p = PRESETS.get(key)
        return p["name"] if p else "未知"

    def list_presets(self) -> dict[str, str]:
        return {k: v["name"] for k, v in PRESETS.items()}

    def forward_kinematics(self, joints: list[float]) -> dict:
        """
        正运动学：6个关节角（弧度）→ EE 位姿。
        返回：{"position":[x,y,z], "rpy_deg":[r,p,y], "matrix":4x4 ndarray}
        """
        T   = self._chain.forward_kinematics(self._to_full_q(joints))
        pos = T[:3, 3].tolist()
        rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True).tolist()
        return {
            "position": [round(v, 4) for v in pos],
            "rpy_deg":  [round(v, 2) for v in rpy],
            "matrix":   T,
        }

    def plan_ee_rotation(
        self,
        current_joints: list[float],
        axis: str,
        angle_deg: float,
        frame: str = "ee",
        max_step_deg: float = 3.0,
        pos_tol_mm: float = 0.5,
    ) -> list[float] | None:
        """
        保持末端 EE 位置不变，绕指定轴旋转。

        参数：
            current_joints : 当前 6 个关节角（弧度）
            axis           : 旋转轴 "x" / "y" / "z"
            angle_deg      : 旋转角度（度），正负决定方向
            frame          : "ee"    = 绕 EE 自身坐标系的轴（默认）
                             "world" = 绕世界坐标系的轴
            max_step_deg   : 每步最大角度（度），控制 IK 步长，默认3度
            pos_tol_mm     : 允许的最大位置误差（毫米），默认0.5mm

        返回：新的 6 个关节角（弧度），IK 失败返回 None
        """
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis 必须是 'x'/'y'/'z'，收到：{axis!r}")

        n_steps  = max(1, int(math.ceil(abs(angle_deg) / max_step_deg)))
        step_deg = angle_deg / n_steps

        q_current = self._to_full_q(current_joints)

        for i in range(n_steps):
            T      = self._chain.forward_kinematics(q_current)
            target_pos = T[:3, 3].copy()
            dR     = Rotation.from_euler(axis, math.radians(step_deg)).as_matrix()
            target_rot = (T[:3, :3] @ dR) if frame == "ee" else (dR @ T[:3, :3])

            q_next, err_mm = self._solve_ik(q_current, target_pos, target_rot)

            if err_mm > pos_tol_mm:
                print(
                    f"[Planner] IK第{i+1}/{n_steps}步失败"
                    f"（误差{err_mm:.2f}mm > 阈值{pos_tol_mm}mm）"
                    f"，可能接近奇异点或超出工作空间"
                )
                return None

            q_current = q_next

        result = list(q_current[1:7])
        print(
            f"[Planner] EE旋转  axis={axis}  angle={angle_deg:.1f}°"
            f"  frame={frame}  {n_steps}步完成"
        )
        return result

    def plan_ee_rotation_steps(
        self,
        current_joints: list[float],
        axis: str,
        total_angle_deg: float,
        n_waypoints: int,
        frame: str = "ee",
        max_step_deg: float = 3.0,
        pos_tol_mm: float = 0.5,
    ) -> list[list[float]] | None:
        """
        生成绕 EE 旋转的多路径点轨迹（每个路径点都是独立发送给控制器的目标）。

        参数：
            n_waypoints     : 路径点数量（均匀分布在总角度内）
            其余参数同 plan_ee_rotation

        返回：路径点列表，每个元素是6个关节角，失败返回 None
        """
        if n_waypoints < 1:
            raise ValueError("n_waypoints 至少为 1")

        angle_per_wp = total_angle_deg / n_waypoints
        trajectory   = []
        q_current    = current_joints

        for i in range(n_waypoints):
            q_next = self.plan_ee_rotation(
                q_current, axis, angle_per_wp, frame, max_step_deg, pos_tol_mm
            )
            if q_next is None:
                print(f"[Planner] 路径点{i+1}/{n_waypoints}规划失败，中止")
                return None
            trajectory.append(q_next)
            q_current = q_next

        print(
            f"[Planner] 多路径点规划完成：{n_waypoints}点  总角度={total_angle_deg:.1f}°"
        )
        return trajectory

    # ═══════════════════════════════════════════
    # 内部：自定义 IK 求解（L-BFGS-B，位置权重高）
    # ═══════════════════════════════════════════

    def _solve_ik(
        self,
        q_init: list[float],
        target_pos: np.ndarray,
        target_rot: np.ndarray,
    ) -> tuple[list[float], float]:
        """
        自定义 IK：以位置误差为主要惩罚项，旋转误差为次要。
        返回 (q_result, 位置误差_mm)
        """
        active_idx = [1, 2, 3, 4, 5, 6]  # q_full 中活跃关节的索引

        def cost(dq: np.ndarray) -> float:
            q = list(q_init)
            for i, idx in enumerate(active_idx):
                q[idx] = dq[i]
            T       = self._chain.forward_kinematics(q)
            pos_err = np.linalg.norm(T[:3, 3] - target_pos)
            rot_err = np.linalg.norm(T[:3, :3] - target_rot)
            # 位置误差权重远大于旋转，确保 EE 位置不漂移
            return (pos_err * 1000.0) ** 2 + rot_err ** 2

        x0     = np.array([q_init[i] for i in active_idx])
        result = minimize(
            cost, x0,
            method="L-BFGS-B",
            bounds=_JOINT_BOUNDS,
            options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-10},
        )

        q_result = list(q_init)
        for i, idx in enumerate(active_idx):
            q_result[idx] = float(result.x[i])

        T_check  = self._chain.forward_kinematics(q_result)
        err_mm   = np.linalg.norm(T_check[:3, 3] - target_pos) * 1000.0
        return q_result, err_mm

    # ═══════════════════════════════════════════
    # 工具
    # ═══════════════════════════════════════════

    @staticmethod
    def _to_full_q(joints6: list[float]) -> list[float]:
        """6关节角 → ikpy 需要的 10 维向量"""
        return [0.0] + list(joints6) + [0.0, 0.0, 0.0]


# ─────────────────────────────────────────────
# 单独测试入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    planner = FR5TrajectoryPlanner()

    print("\n=== 预设位置 ===")
    for k, name in planner.list_presets().items():
        print(f"  {k}: {name}  {planner.get_preset_joints(k)}")

    print("\n=== 正运动学（预设位置1）===")
    j1 = planner.get_preset_joints("1")
    fk = planner.forward_kinematics(j1)
    print(f"  EE位置(m):  {fk['position']}")
    print(f"  EE姿态(°):  {fk['rpy_deg']}")

    print("\n=== 绕EE Z轴旋转90度 ===")
    j_rot = planner.plan_ee_rotation(j1, axis="z", angle_deg=90.0)
    if j_rot:
        fk2 = planner.forward_kinematics(j_rot)
        print(f"  旋转后EE位置: {fk2['position']}  （应与旋转前相同）")
        print(f"  旋转后EE姿态: {fk2['rpy_deg']}")

    print("\n=== 绕EE X轴旋转45度，分3个路径点 ===")
    traj = planner.plan_ee_rotation_steps(j1, axis="x", total_angle_deg=45.0, n_waypoints=3)
    if traj:
        for i, q in enumerate(traj):
            fk_i = planner.forward_kinematics(q)
            print(f"  路径点{i+1}: pos={fk_i['position']}")
