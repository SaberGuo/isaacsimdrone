# omniperception_isaacdrone/controller/lee_position_controller.py

from __future__ import annotations

import math
import torch
import torch.nn as nn


def _normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / (torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps))


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


def quat_to_rotmat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """q: (...,4) in (w,x,y,z) -> R: (...,3,3)"""
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = torch.stack([
        torch.stack([ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy)  ], dim=-1),
        torch.stack([2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx)  ], dim=-1),
        torch.stack([2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz], dim=-1),
    ], dim=-2)
    return R


def quat_to_yaw_wxyz(q: torch.Tensor) -> torch.Tensor:
    R = quat_to_rotmat_wxyz(q)
    return torch.atan2(R[..., 1, 0], R[..., 0, 0])


def world_to_body_vec(q_wxyz: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
    R = quat_to_rotmat_wxyz(q_wxyz)
    return torch.matmul(R.transpose(-2, -1), v_w.unsqueeze(-1)).squeeze(-1)


class LeeVelocityYawRateController(nn.Module):
    """Lee-style controller (customized):
    - Track desired velocity in world frame (vx,vy,vz)
    - Track desired yaw (integrated from yaw_rate outside, passed as target_yaw)
    - Track desired yaw_rate (body frame)

    Outputs:
      thrust: (N,1)  along body +Z
      torque_body: (N,3) in body frame
    """

    def __init__(
        self,
        g: float = 9.81,
        vel_gain: tuple[float, float, float] = (3.0, 3.0, 4.0),
        pos_gain: tuple[float, float, float] = (0.0, 0.0, 0.0),
        attitude_gain: tuple[float, float, float] = (6.0, 6.0, 1.5),
        ang_rate_gain: tuple[float, float, float] = (0.25, 0.25, 0.18),
        mass: float | torch.Tensor = 1.0,
        inertia_diag: tuple[float, float, float] | torch.Tensor = (0.02, 0.02, 0.04),
        thrust_limit_factor: float = 3.0,
        torque_limit: tuple[float, float, float] = (200.0, 200.0, 200.0),
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        self.g = float(g)

        self.pos_gain      = torch.tensor(pos_gain,       dtype=torch.float32, device=device).view(1, 3)
        self.vel_gain      = torch.tensor(vel_gain,       dtype=torch.float32, device=device).view(1, 3)
        self.attitude_gain = torch.tensor(attitude_gain,  dtype=torch.float32, device=device).view(1, 3)
        self.ang_rate_gain = torch.tensor(ang_rate_gain,  dtype=torch.float32, device=device).view(1, 3)
        self.torque_limit  = torch.tensor(torque_limit,   dtype=torch.float32, device=device).view(1, 3)

        self.thrust_limit_factor = float(thrust_limit_factor)

        self.register_buffer("mass",         torch.as_tensor(mass,         dtype=torch.float32, device=device))
        self.register_buffer("inertia_diag", torch.as_tensor(inertia_diag, dtype=torch.float32, device=device))

    def forward(
        self,
        root_state_w: torch.Tensor,           # (N,13): pos(3), quat_wxyz(4), lin_vel_w(3), ang_vel_w(3)
        target_vel_w: torch.Tensor,           # (N,3)
        target_yaw: torch.Tensor,             # (N,) or (N,1)
        target_yaw_rate: torch.Tensor | None = None,
        target_pos_w: torch.Tensor | None = None,
        target_acc_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = root_state_w.device
        N = root_state_w.shape[0]

        pos       = root_state_w[:, 0:3]
        quat      = root_state_w[:, 3:7]
        vel       = root_state_w[:, 7:10]
        ang_vel_w = root_state_w[:, 10:13]

        # --- reshape scalars ---
        if target_yaw.dim() == 2 and target_yaw.shape[1] == 1:
            target_yaw = target_yaw[:, 0]
        target_yaw = target_yaw.reshape(N)

        if target_yaw_rate is None:
            target_yaw_rate = torch.zeros(N, device=device, dtype=torch.float32)
        elif target_yaw_rate.dim() == 2 and target_yaw_rate.shape[1] == 1:
            target_yaw_rate = target_yaw_rate[:, 0]
        target_yaw_rate = target_yaw_rate.reshape(N)

        if target_pos_w is None:
            target_pos_w = pos
        if target_acc_w is None:
            target_acc_w = torch.zeros(N, 3, device=device, dtype=torch.float32)

        # --- broadcast mass (N,1) and inertia (N,3) ---
        m = self.mass.to(device)
        if m.dim() == 0:
            m = m.view(1, 1).expand(N, 1)
        elif m.dim() == 1:
            m = m.view(-1, 1)
            if m.shape[0] == 1:
                m = m.expand(N, 1)

        Jd = self.inertia_diag.to(device)
        if Jd.dim() == 1 and Jd.shape[0] == 3:
            Jd = Jd.view(1, 3).expand(N, 3)
        elif Jd.dim() == 2 and Jd.shape[0] == 1:
            Jd = Jd.expand(N, 3)

        Kp = self.pos_gain.to(device)
        Kv = self.vel_gain.to(device)
        KR = self.attitude_gain.to(device)
        Kw = self.ang_rate_gain.to(device)

        # gravity vector pointing DOWN (world frame)
        g_vec = torch.tensor([0.0, 0.0, self.g], device=device, dtype=torch.float32).view(1, 3)

        pos_error = pos - target_pos_w
        vel_error = vel - target_vel_w

        # acc 一般指向下方（含重力项）
        acc = Kp * pos_error + Kv * vel_error - g_vec - target_acc_w  # (N,3)

        # 期望机体 Z 轴指向上方
        b3_des = -_normalize(acc)  # (N,3)

        # 当前旋转矩阵
        R  = quat_to_rotmat_wxyz(quat)  # (N,3,3)
        b3 = R[:, :, 2]                  # (N,3) 当前机体 Z 轴

        # thrust = -m * (acc · b3)，acc 指下 b3 指上，点积<0，双负得正
        # 先保存 pre-clamp 值用于调试
        acc_dot_b3    = (acc * b3).sum(dim=-1, keepdim=True)       # (N,1)
        thrust_raw    = -m * acc_dot_b3                             # (N,1) pre-clamp

        thrust_max    = (self.thrust_limit_factor * m * self.g).clamp_min(1e-3)
        thrust_min    = torch.zeros_like(thrust_raw)
        thrust        = torch.clamp(thrust_raw, min=thrust_min, max=thrust_max)

        # 期望 yaw 对应的 b1_des
        b1_des = torch.stack([
            torch.cos(target_yaw),
            torch.sin(target_yaw),
            torch.zeros_like(target_yaw),
        ], dim=-1)  # (N,3)

        # 构造期望旋转矩阵
        b2_des = _normalize(torch.cross(b3_des, b1_des, dim=-1))
        b1     = torch.cross(b2_des, b3_des, dim=-1)
        R_des  = torch.stack([b1, b2_des, b3_des], dim=-1)  # (N,3,3)

        # 姿态误差
        RtR = torch.matmul(R_des.transpose(1, 2), R)
        Rtr = torch.matmul(R.transpose(1, 2), R_des)
        E   = RtR - Rtr
        e_R = 0.5 * torch.stack([E[:, 2, 1], E[:, 0, 2], E[:, 1, 0]], dim=-1)  # (N,3)

        # 角速度转到机体系
        omega_body = world_to_body_vec(quat, ang_vel_w)  # (N,3)

        # 期望角速度（仅 yaw rate）
        omega_des_body = torch.zeros_like(omega_body)
        omega_des_body[:, 2] = target_yaw_rate
        e_omega = omega_body - omega_des_body

        # 力矩
        Jomega        = omega_body * Jd
        coriolis      = torch.cross(omega_body, Jomega, dim=-1)
        torque_raw    = -KR * e_R - Kw * e_omega + coriolis        # pre-clamp
        torque        = torch.clamp(torque_raw, min=-self.torque_limit.to(device), max=self.torque_limit.to(device))

        # =====================================================================
        # DEBUG 打印（所有过程量，放在最后便于一键注释）
        # =====================================================================
        # print("=" * 60)
        # print(f"[Lee] pos               : {pos}")
        # print(f"[Lee] vel               : {vel}")
        # print(f"[Lee] quat(wxyz)        : {quat}")
        # print(f"[Lee] target_vel_w      : {target_vel_w}")
        # print(f"[Lee] target_yaw        : {target_yaw}")
        # print(f"[Lee] target_yaw_rate   : {target_yaw_rate}")
        # print(f"[Lee] m                 : {m}")
        # print(f"[Lee] Jd (inertia diag) : {Jd[0]}")
        # print(f"[Lee] g_vec             : {g_vec}")
        # print(f"[Lee] pos_error         : {pos_error}")
        # print(f"[Lee] vel_error         : {vel_error}")
        # print(f"[Lee] acc               : {acc}")
        # print(f"[Lee] b3_des            : {b3_des}")
        # print(f"[Lee] b3 (current)      : {b3}")
        # print(f"[Lee] acc·b3 (dot)      : {acc_dot_b3.squeeze(-1)}")
        # print(f"[Lee] thrust_raw (pre-clamp): {thrust_raw.squeeze(-1)}")
        # print(f"[Lee] thrust_min        : {thrust_min.squeeze(-1)}")
        # print(f"[Lee] thrust_max        : {thrust_max.squeeze(-1)}")
        # print(f"[Lee] thrust (final)    : {thrust.squeeze(-1)}")
        # print(f"[Lee] b1_des            : {b1_des}")
        # print(f"[Lee] b2_des            : {b2_des}")
        # print(f"[Lee] b3_des (R_des col): {R_des[:, :, 2]}")
        # print(f"[Lee] e_R               : {e_R}")
        # print(f"[Lee] omega_body        : {omega_body}")
        # print(f"[Lee] omega_des_body    : {omega_des_body}")
        # print(f"[Lee] e_omega           : {e_omega}")
        # print(f"[Lee] Jomega            : {Jomega}")
        # print(f"[Lee] coriolis          : {coriolis}")
        # print(f"[Lee] torque_raw (pre-clamp): {torque_raw}")
        # print(f"[Lee] torque (final)    : {torque}")
        # print("=" * 60)
        # =====================================================================

        return thrust, torque
