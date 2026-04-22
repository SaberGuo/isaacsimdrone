from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for LeePositionController config loading.") from exc


def _normalize(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)


def quat_to_rotmat_wxyz(q: torch.Tensor) -> torch.Tensor:
    """q: (..., 4) [w, x, y, z] -> (..., 3, 3)"""
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    return torch.stack(
        [
            torch.stack([ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)], dim=-1),
            torch.stack([2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)], dim=-1),
            torch.stack([2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz], dim=-1),
        ],
        dim=-2,
    )


def quat_rotate_inverse_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r = quat_to_rotmat_wxyz(q)
    return torch.matmul(r.transpose(-2, -1), v.unsqueeze(-1)).squeeze(-1)


def _quat_to_yaw_wxyz(q: torch.Tensor) -> torch.Tensor:
    r = quat_to_rotmat_wxyz(q)
    return torch.atan2(r[..., 1, 0], r[..., 0, 0])


def compute_parameters(rotor_config: dict, inertia_matrix_4x4: torch.Tensor) -> torch.Tensor:
    rotor_angles = torch.as_tensor(rotor_config["rotor_angles"], dtype=torch.float32)
    arm_lengths = torch.as_tensor(rotor_config["arm_lengths"], dtype=torch.float32)
    force_constants = torch.as_tensor(rotor_config["force_constants"], dtype=torch.float32)
    moment_constants = torch.as_tensor(rotor_config["moment_constants"], dtype=torch.float32)
    directions = torch.as_tensor(rotor_config["directions"], dtype=torch.float32)

    a = torch.stack(
        [
            torch.sin(rotor_angles) * arm_lengths,
            -torch.cos(rotor_angles) * arm_lengths,
            -directions * moment_constants / force_constants,
            torch.ones_like(rotor_angles),
        ]
    )
    return a.t() @ torch.linalg.inv(a @ a.t()) @ inertia_matrix_4x4


class LeePositionController(nn.Module):
    """Lee controller ported from OmniPerception/OmniDrones.

    Output is normalized rotor command in [-1, 1] for each rotor.
    """

    def __init__(self, g: float, uav_params: dict, controller_param_path: str | None = None) -> None:
        super().__init__()

        uav_name = str(uav_params.get("name", "iris"))
        if controller_param_path is None:
            controller_param_path = (
                Path(__file__).resolve().parent / "cfg" / f"lee_controller_{uav_name}.yaml"
            )

        with open(controller_param_path, "r", encoding="utf-8") as f:
            controller_params = yaml.safe_load(f)

        rotor_config = uav_params["rotor_configuration"]
        inertia = uav_params["inertia"]

        self.pos_gain = nn.Parameter(torch.as_tensor(controller_params["position_gain"], dtype=torch.float32))
        self.vel_gain = nn.Parameter(torch.as_tensor(controller_params["velocity_gain"], dtype=torch.float32))
        self.mass = nn.Parameter(torch.tensor(float(uav_params["mass"]), dtype=torch.float32))
        self.g = nn.Parameter(torch.tensor([0.0, 0.0, abs(float(g))], dtype=torch.float32))

        force_constants = torch.as_tensor(rotor_config["force_constants"], dtype=torch.float32)
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"], dtype=torch.float32)
        self.max_thrusts = nn.Parameter(max_rot_vel.square() * force_constants)

        i = torch.diag_embed(
            torch.tensor(
                [
                    float(inertia["xx"]),
                    float(inertia["yy"]),
                    float(inertia["zz"]),
                    1.0,
                ],
                dtype=torch.float32,
            )
        )
        self.mixer = nn.Parameter(compute_parameters(rotor_config, i))
        self.attitute_gain = nn.Parameter(
            torch.as_tensor(controller_params["attitude_gain"], dtype=torch.float32) @ torch.linalg.inv(i[:3, :3])
        )
        self.ang_rate_gain = nn.Parameter(
            torch.as_tensor(controller_params["angular_rate_gain"], dtype=torch.float32)
            @ torch.linalg.inv(i[:3, :3])
        )

        self.requires_grad_(False)

    def process_rl_actions(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_vel, target_yaw_rate = actions.split([3, 1], dim=-1)
        return target_vel, target_yaw_rate

    def compute(
        self,
        root_state: torch.Tensor,
        target_pos: torch.Tensor | None = None,
        target_vel: torch.Tensor | None = None,
        target_acc: torch.Tensor | None = None,
        target_yaw_rate: torch.Tensor | None = None,
        body_rate: bool = False,
    ) -> torch.Tensor:
        batch_shape = root_state.shape[:-1]
        device = root_state.device

        if target_pos is None:
            target_pos = root_state[..., :3]
        else:
            target_pos = target_pos.expand(batch_shape + (3,))

        if target_vel is None:
            target_vel = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_vel = target_vel.expand(batch_shape + (3,))

        if target_acc is None:
            target_acc = torch.zeros(*batch_shape, 3, device=device)
        else:
            target_acc = target_acc.expand(batch_shape + (3,))

        if target_yaw_rate is None:
            target_yaw_rate = torch.zeros(*batch_shape, 1, device=device)
        else:
            if target_yaw_rate.shape[-1] != 1:
                target_yaw_rate = target_yaw_rate.unsqueeze(-1)
            target_yaw_rate = target_yaw_rate.expand(batch_shape + (1,))

        cmd = self._compute(
            root_state.reshape(-1, 13),
            target_pos.reshape(-1, 3),
            target_vel.reshape(-1, 3),
            target_acc.reshape(-1, 3),
            target_yaw_rate.reshape(-1, 1),
            body_rate,
        )
        return cmd.reshape(*batch_shape, -1)

    def _compute(
        self,
        root_state: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        target_acc: torch.Tensor,
        target_yaw_rate: torch.Tensor,
        body_rate: bool,
    ) -> torch.Tensor:
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        if not body_rate:
            ang_vel = quat_rotate_inverse_wxyz(rot, ang_vel)

        pos_error = pos - target_pos
        vel_error = vel - target_vel

        acc = pos_error * self.pos_gain + vel_error * self.vel_gain - self.g - target_acc

        r = quat_to_rotmat_wxyz(rot)
        current_yaw = _quat_to_yaw_wxyz(rot).unsqueeze(-1)
        b1_des = torch.cat(
            [
                torch.cos(current_yaw),
                torch.sin(current_yaw),
                torch.zeros_like(current_yaw),
            ],
            dim=-1,
        )
        b3_des = -_normalize(acc)
        b2_des = _normalize(torch.cross(b3_des, b1_des, dim=1))
        r_des = torch.stack(
            [
                torch.cross(b2_des, b3_des, dim=1),
                b2_des,
                b3_des,
            ],
            dim=-1,
        )

        ang_error_matrix = 0.5 * (
            torch.bmm(r_des.transpose(-2, -1), r) - torch.bmm(r.transpose(-2, -1), r_des)
        )
        ang_error = torch.stack(
            [
                ang_error_matrix[:, 2, 1],
                ang_error_matrix[:, 0, 2],
                ang_error_matrix[:, 1, 0],
            ],
            dim=-1,
        )

        angular_rate_des = torch.zeros_like(ang_vel)
        angular_rate_des[:, 2] = target_yaw_rate.squeeze(1)
        ang_rate_err = ang_vel - torch.bmm(
            torch.bmm(r_des.transpose(-2, -1), r),
            angular_rate_des.unsqueeze(2),
        ).squeeze(2)
        ang_acc = -ang_error * self.attitute_gain - ang_rate_err * self.ang_rate_gain + torch.linalg.cross(
            ang_vel, ang_vel
        )

        thrust = -self.mass * (acc * r[:, :, 2]).sum(-1, keepdim=True)
        ang_acc_thrust = torch.cat([ang_acc, thrust], dim=-1)

        cmd = (self.mixer @ ang_acc_thrust.t()).t()
        cmd = (cmd / self.max_thrusts) * 2.0 - 1.0
        return torch.clamp(cmd, -1.0, 1.0)


# compatibility alias
LeeVelocityYawRateController = LeePositionController
