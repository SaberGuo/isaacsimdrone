from __future__ import annotations

import math
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTermCfg as ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm

from omniperception_isaacdrone.controller import LeeVelocityYawRateController


def _wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return (x + math.pi) % (2 * math.pi) - math.pi


def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """quat: (N,4) [w,x,y,z] -> yaw (N,)"""
    w, x, y, z = quat.unbind(-1)
    r00 = w * w + x * x - y * y - z * z
    r10 = 2.0 * (x * y + w * z)
    return torch.atan2(r10, r00)


def _get_step_dt(env: ManagerBasedRLEnv) -> float:
    """Return RL step dt (sim.dt * decimation), robustly."""
    if hasattr(env, "step_dt"):
        try:
            return float(env.step_dt)
        except Exception:
            pass
    try:
        dt = float(env.cfg.sim.dt)
        dec = float(getattr(env.cfg, "decimation", 1))
        return dt * dec
    except Exception:
        return 1.0 / 60.0


class RootTwistVelocityActionTerm(ActionTerm):
    """4D 动作: [delta_vx, delta_vy, delta_vz, yaw_rate]
    - delta_vx, delta_vy, delta_vz: world frame velocity change (m/s per step)
    - yaw_rate: desired yaw rate in body frame (rad/s)

    动作解释：
    - 前三个维度表示相对于当前目标速度的变化量（增量控制）
    - yaw_rate 仍然是绝对角速度命令（积分得到目标偏航角）
    - 目标速度会通过 clip 限制在 [-vel_clip, vel_clip] 范围内

    修复点：
    - 强制 raw action clip 到 [-1, 1]，保证 action_space 有意义且 PPO 稳定
    - 使用增量控制使动作更加平滑，便于学习精细控制
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]
        self._device = env.device
        self._num_envs = env.num_envs

        self._dt = _get_step_dt(env)

        self._raw_actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self._processed_actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)

        self._forces = torch.zeros((self._num_envs, 1, 3), device=self._device, dtype=torch.float32)
        self._torques = torch.zeros((self._num_envs, 1, 3), device=self._device, dtype=torch.float32)

        self._yaw_target = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)

        # 目标速度缓冲区（增量控制使用）
        self._target_vel = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)

        params = getattr(cfg, "params", None) or {}
        p_get = params.get if isinstance(params, dict) else lambda k, d=None: getattr(params, k, d)

        self._vel_scale = float(p_get("vel_scale", 4.0))
        self._vel_clip = float(p_get("vel_clip", 5.0))
        self._yaw_rate_scale = float(p_get("yaw_rate_scale", 2.0))
        self._yaw_rate_clip = float(p_get("yaw_rate_clip", 3.0))
        self._thrust_sign = float(p_get("thrust_sign", 1.0))

        self._g = float(p_get("g", 9.81))
        self._vel_gain = tuple(p_get("vel_gain", (3.0, 3.0, 4.0)))
        self._pos_gain = tuple(p_get("pos_gain", (0.0, 0.0, 0.0)))
        self._attitude_gain = tuple(p_get("attitude_gain", (6.0, 6.0, 1.5)))
        self._ang_rate_gain = tuple(p_get("ang_rate_gain", (0.25, 0.25, 0.18)))
        self._thrust_limit_factor = float(p_get("thrust_limit_factor", 3.0))
        self._torque_limit = tuple(p_get("torque_limit", (200.0, 200.0, 200.0)))

        self._use_sim_total_mass = bool(p_get("use_sim_total_mass", True))
        self._prevent_negative_thrust = bool(p_get("prevent_negative_thrust", True))

        desired_total_mass = float(p_get("mass", 0.29))

        inertia_diag = tuple(p_get("inertia_diag", (0.02, 0.02, 0.04)))
        self._inertia_diag = (
            torch.tensor(inertia_diag, device=self._device, dtype=torch.float32)
            .view(1, 3)
            .expand(self._num_envs, 3)
            .contiguous()
        )

        body_id = 0
        try:
            ids, _ = self._asset.find_bodies(["base", "base_link", ".*base.*", "body", ".*body.*"], preserve_order=True)
            if len(ids) > 0:
                body_id = int(ids[0])
        except Exception:
            pass
        self._body_ids = [body_id]

        print("body names:", self._asset.body_names)
        print("selected body_id:", self._body_ids[0], "name:", self._asset.body_names[self._body_ids[0]])

        sim_total_mass = None
        sim_body_masses = None
        try:
            sim_body_masses = getattr(self._asset.data, "default_mass", None)
            if sim_body_masses is not None:
                sim_total_mass = sim_body_masses.sum(dim=1, keepdim=True).to(torch.float32)
                if sim_total_mass.device != self._device:
                    sim_total_mass = sim_total_mass.to(self._device)
        except Exception:
            sim_total_mass = None

        if sim_total_mass is None:
            sim_total_mass = torch.full((self._num_envs, 1), desired_total_mass, device=self._device, dtype=torch.float32)

        sim_total0 = float(sim_total_mass[0, 0].item())

        try:
            if sim_body_masses is not None:
                masses0 = sim_body_masses[0].detach().cpu().numpy().tolist()
                pairs = [f"{n}:{m:.4f}" for n, m in zip(self._asset.body_names, masses0)]
                print(f"[root_twist] Sim body masses (kg): {', '.join(pairs)}", flush=True)
        except Exception:
            pass

        print(f"[root_twist] Desired TOTAL mass (cfg.params.mass): {desired_total_mass:.4f} kg", flush=True)
        print(f"[root_twist] Sim TOTAL mass (sum bodies):         {sim_total0:.4f} kg", flush=True)
        print(f"[root_twist] Hover thrust needed (sim):          {sim_total0 * self._g:.3f} N", flush=True)
        print(f"[root_twist] RL step_dt used for yaw integration: {self._dt:.6f} s", flush=True)

        rel_err = abs(sim_total0 - desired_total_mass) / max(abs(desired_total_mass), 1e-6)
        if rel_err > 0.05:
            print(
                f"[WARN][root_twist] Mass mismatch detected! cfg.mass={desired_total_mass:.4f} kg "
                f"but sim total mass={sim_total0:.4f} kg. "
                "This is commonly caused by UsdFileCfg.mass_props being applied to EVERY link of an articulation.",
                flush=True,
            )

        if self._use_sim_total_mass:
            self._mass = sim_total_mass
        else:
            self._mass = torch.full((self._num_envs, 1), desired_total_mass, device=self._device, dtype=torch.float32)

        self._controller = LeeVelocityYawRateController(
            g=self._g,
            vel_gain=self._vel_gain,
            pos_gain=self._pos_gain,
            attitude_gain=self._attitude_gain,
            ang_rate_gain=self._ang_rate_gain,
            mass=self._mass.squeeze(-1),
            inertia_diag=self._inertia_diag,
            thrust_limit_factor=self._thrust_limit_factor,
            torque_limit=self._torque_limit,
            device=self._device,
        ).to(self._device)
        self._controller.eval()

        self._init_yaw_target_all()

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _init_yaw_target_all(self):
        try:
            quat = self._asset.data.root_link_state_w[:, 3:7]
            self._yaw_target.copy_(_quat_to_yaw(quat))
        except Exception:
            self._yaw_target.zero_()

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._forces.zero_()
            self._torques.zero_()
            self._target_vel.zero_()  # 重置目标速度
            self._init_yaw_target_all()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._forces[env_ids] = 0.0
            self._torques[env_ids] = 0.0
            self._target_vel[env_ids] = 0.0  # 重置目标速度
            try:
                quat = self._asset.data.root_link_state_w[env_ids, 3:7]
                self._yaw_target[env_ids] = _quat_to_yaw(quat)
            except Exception:
                self._yaw_target[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        # --- enforce raw action normalization strictly ---
        if actions.device != self._device:
            actions = actions.to(self._device)
        actions = actions.to(torch.float32)

        # IMPORTANT: clamp raw action into [-1, 1]
        actions = torch.clamp(actions, -1.0, 1.0)

        # store clipped raw actions
        self._raw_actions.copy_(actions)

        # 动作解释：前三个维度是速度变化量（增量控制），第四个是偏航角速度
        delta_v = actions[:, 0:3] * self._vel_scale  # 速度变化量
        yaw_rate_cmd = actions[:, 3:4] * self._yaw_rate_scale

        # 更新目标速度（累加变化量）
        self._target_vel += delta_v

        # 对目标速度进行限幅
        if self._vel_clip > 0:
            self._target_vel = torch.clamp(self._target_vel, -self._vel_clip, self._vel_clip)
        if self._yaw_rate_clip > 0:
            yaw_rate_cmd = torch.clamp(yaw_rate_cmd, -self._yaw_rate_clip, self._yaw_rate_clip)

        self._processed_actions[:, 0:3] = self._target_vel
        self._processed_actions[:, 3:4] = yaw_rate_cmd

    def apply_actions(self):
        yaw_rate_cmd = self._processed_actions[:, 3]
        self._yaw_target = _wrap_to_pi(self._yaw_target + yaw_rate_cmd * self._dt)

        root = self._asset.data.root_link_state_w
        v_cmd = self._processed_actions[:, 0:3]

        if self._prevent_negative_thrust:
            kv_z = float(self._vel_gain[2])
            if kv_z > 1e-6:
                vz = root[:, 9]
                min_target_vz = vz - (self._g / kv_z)
                if torch.any(v_cmd[:, 2] < min_target_vz):
                    v_cmd = v_cmd.clone()
                    v_cmd[:, 2] = torch.maximum(v_cmd[:, 2], min_target_vz)

        thrust, torque_body = self._controller(
            root_state_w=root,
            target_vel_w=v_cmd,
            target_yaw=self._yaw_target,
            target_yaw_rate=yaw_rate_cmd,
            target_pos_w=None,
            target_acc_w=None,
        )

        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, 0, 2] = self._thrust_sign * thrust.squeeze(-1)
        self._torques[:, 0, :] = torque_body

        if hasattr(self._asset, "set_external_force_and_torque"):
            self._asset.set_external_force_and_torque(
                forces=self._forces,
                torques=self._torques,
                body_ids=self._body_ids,
                env_ids=None,
            )
        else:
            # Backward compatibility for older IsaacLab API variants.
            self._asset.permanent_wrench_composer.set_forces_and_torques(
                forces=self._forces,
                torques=self._torques,
                positions=None,
                body_ids=self._body_ids,
                env_ids=None,
                is_global=False,
            )
