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


class RootTwistVelocityActionTerm(ActionTerm):
    """4D 动作: [vx, vy, vz, yaw_rate]
    - vx,vy,vz: world frame desired velocity (m/s)
    - yaw_rate: desired yaw rate in body frame (rad/s)

    关键修复点：
    1) 自动读取仿真里的“总质量”（sum over bodies），避免 cfg.mass 与仿真质量不一致导致推力不足。
    2) 可选：根据当前 vz 和 Kv_z，限制 target_vz 下界，避免要求“负推力”（会被 clamp 成 0）。
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset    = env.scene[cfg.asset_name]
        self._device   = env.device
        self._num_envs = env.num_envs
        self._dt       = float(getattr(env, "physics_dt", env.cfg.sim.dt))

        # action buffers
        self._raw_actions       = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self._processed_actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)

        # wrench buffers (num_envs, 1_body, 3)
        self._forces  = torch.zeros((self._num_envs, 1, 3), device=self._device, dtype=torch.float32)
        self._torques = torch.zeros((self._num_envs, 1, 3), device=self._device, dtype=torch.float32)

        # yaw target buffer
        self._yaw_target = torch.zeros((self._num_envs,), device=self._device, dtype=torch.float32)

        # ------------------------------------------------------------------ #
        #  cfg.params
        # ------------------------------------------------------------------ #
        params = getattr(cfg, "params", None) or {}
        p_get  = params.get if isinstance(params, dict) else lambda k, d=None: getattr(params, k, d)

        self._vel_scale            = float(p_get("vel_scale",           3.0))
        self._vel_clip             = float(p_get("vel_clip",            5.0))
        self._yaw_rate_scale       = float(p_get("yaw_rate_scale",      2.0))
        self._yaw_rate_clip        = float(p_get("yaw_rate_clip",       3.0))
        self._thrust_sign          = float(p_get("thrust_sign",         1.0))

        self._g                    = float(p_get("g",                   9.81))
        self._vel_gain             = tuple(p_get("vel_gain",            (3.0, 3.0, 4.0)))
        self._pos_gain             = tuple(p_get("pos_gain",            (0.0, 0.0, 0.0)))
        self._attitude_gain        = tuple(p_get("attitude_gain",       (6.0, 6.0, 1.5)))
        self._ang_rate_gain        = tuple(p_get("ang_rate_gain",       (0.25, 0.25, 0.18)))
        self._thrust_limit_factor  = float(p_get("thrust_limit_factor", 3.0))
        self._torque_limit         = tuple(p_get("torque_limit",        (200.0, 200.0, 200.0)))

        # 新增：是否使用仿真“总质量”作为控制器质量（推荐 True）
        self._use_sim_total_mass = bool(p_get("use_sim_total_mass", True))

        # 新增：是否限制 target_vz 下界，避免出现“需要负推力”
        self._prevent_negative_thrust = bool(p_get("prevent_negative_thrust", True))

        # cfg 中“期望总质量”（注意：对 articulation 来说，它应该是 TOTAL mass）
        desired_total_mass = float(p_get("mass", 0.29))

        # inertia（仍然用你给的简化值；也可后续从 USD/PhysX 读取更精确）
        inertia_diag = tuple(p_get("inertia_diag", (0.02, 0.02, 0.04)))
        self._inertia_diag = (
            torch.tensor(inertia_diag, device=self._device, dtype=torch.float32)
            .view(1, 3)
            .expand(self._num_envs, 3)
            .contiguous()
        )

        # ------------------------------------------------------------------ #
        #  选取 base body
        # ------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------ #
        #  自动读取仿真总质量（sum over bodies）
        # ------------------------------------------------------------------ #
        sim_total_mass = None
        sim_body_masses = None
        try:
            sim_body_masses = getattr(self._asset.data, "default_mass", None)
            if sim_body_masses is not None:
                # (num_envs, num_bodies) -> total (num_envs, 1)
                sim_total_mass = sim_body_masses.sum(dim=1, keepdim=True).to(torch.float32)
                if sim_total_mass.device != self._device:
                    sim_total_mass = sim_total_mass.to(self._device)
        except Exception:
            sim_total_mass = None

        if sim_total_mass is None:
            # fallback: 使用 cfg 的质量
            sim_total_mass = torch.full(
                (self._num_envs, 1), desired_total_mass, device=self._device, dtype=torch.float32
            )

        sim_total0 = float(sim_total_mass[0, 0].item())

        # 打印一次诊断信息（强烈建议你先看这里确认问题）
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

        rel_err = abs(sim_total0 - desired_total_mass) / max(abs(desired_total_mass), 1e-6)
        if rel_err > 0.05:
            print(
                f"[WARN][root_twist] Mass mismatch detected! cfg.mass={desired_total_mass:.4f} kg "
                f"but sim total mass={sim_total0:.4f} kg. "
                "This is commonly caused by UsdFileCfg.mass_props being applied to EVERY link of an articulation.",
                flush=True,
            )

        # 控制器使用的质量（默认用仿真总质量，最鲁棒）
        if self._use_sim_total_mass:
            self._mass = sim_total_mass
        else:
            self._mass = torch.full(
                (self._num_envs, 1), desired_total_mass, device=self._device, dtype=torch.float32
            )

        # ------------------------------------------------------------------ #
        #  构造 Lee 控制器
        # ------------------------------------------------------------------ #
        self._controller = LeeVelocityYawRateController(
            g=self._g,
            vel_gain=self._vel_gain,
            pos_gain=self._pos_gain,
            attitude_gain=self._attitude_gain,
            ang_rate_gain=self._ang_rate_gain,
            mass=self._mass.squeeze(-1),          # (N,)
            inertia_diag=self._inertia_diag,      # (N,3)
            thrust_limit_factor=self._thrust_limit_factor,
            torque_limit=self._torque_limit,
            device=self._device,
        ).to(self._device)
        self._controller.eval()

        self._init_yaw_target_all()

    # ===================================================================== #
    #  ActionTerm 接口
    # ===================================================================== #

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
            self._init_yaw_target_all()
        else:
            self._raw_actions[env_ids]       = 0.0
            self._processed_actions[env_ids] = 0.0
            self._forces[env_ids]            = 0.0
            self._torques[env_ids]           = 0.0
            try:
                quat = self._asset.data.root_link_state_w[env_ids, 3:7]
                self._yaw_target[env_ids] = _quat_to_yaw(quat)
            except Exception:
                self._yaw_target[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        if actions.device != self._device:
            actions = actions.to(self._device)

        self._raw_actions.copy_(actions)

        v_cmd        = actions[:, 0:3] * self._vel_scale
        yaw_rate_cmd = actions[:, 3:4] * self._yaw_rate_scale

        if self._vel_clip > 0:
            v_cmd = torch.clamp(v_cmd, -self._vel_clip, self._vel_clip)
        if self._yaw_rate_clip > 0:
            yaw_rate_cmd = torch.clamp(yaw_rate_cmd, -self._yaw_rate_clip, self._yaw_rate_clip)

        self._processed_actions[:, 0:3] = v_cmd
        self._processed_actions[:, 3:4] = yaw_rate_cmd

    def apply_actions(self):
        yaw_rate_cmd = self._processed_actions[:, 3]
        self._yaw_target = _wrap_to_pi(self._yaw_target + yaw_rate_cmd * self._dt)

        root  = self._asset.data.root_link_state_w
        v_cmd = self._processed_actions[:, 0:3]

        # ------------------------------------------------------------------ #
        # 可选：避免出现“需要负推力”（downward accel > g）
        # 推导（近似）：
        #   a_total_des_z ≈ Kv_z * (target_vz - vz) + g
        # 需要 thrust >= 0  => a_total_des_z >= 0
        # => target_vz >= vz - g / Kv_z
        # ------------------------------------------------------------------ #
        if self._prevent_negative_thrust:
            kv_z = float(self._vel_gain[2])
            if kv_z > 1e-6:
                vz = root[:, 9]  # root_state_w: lin_vel_w is [7:10], so z is index 9
                min_target_vz = vz - (self._g / kv_z)
                # only clone if we need to modify
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
        self._forces[:, 0, 2]  = self._thrust_sign * thrust.squeeze(-1)
        self._torques[:, 0, :] = torque_body

        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=self._forces,
            torques=self._torques,
            positions=None,
            body_ids=self._body_ids,
            env_ids=None,
            is_global=False,
        )
