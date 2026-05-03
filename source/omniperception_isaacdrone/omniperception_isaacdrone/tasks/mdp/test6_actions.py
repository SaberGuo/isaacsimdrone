"""Action terms for the Test6 drone task."""

from __future__ import annotations

import math
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ActionTermCfg as ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm

from omniperception_isaacdrone.actuator import RotorGroup
from omniperception_isaacdrone.assets.robots.drone_cfg import DRONE_PARAMS
from omniperception_isaacdrone.controller import LeePositionController


# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------
def _get_step_dt(env: ManagerBasedRLEnv) -> float:
    if hasattr(env, "step_dt"):
        try:
            return float(env.step_dt)
        except Exception:
            pass
    try:
        return float(env.cfg.sim.dt) * float(getattr(env.cfg, "decimation", 1))
    except Exception:
        return 1.0 / 60.0


# -----------------------------------------------------------------------------
# Action terms
# -----------------------------------------------------------------------------
class RootTwistVelocityActionTerm(ActionTerm):
    """OmniPerception-style controller + actuator chain.

    4D action: [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
      - velocity command is mapped to target velocity in world frame
      - yaw rate command is mapped to target yaw rate in rad/s

    Pipeline:
      rl action -> LeePositionController (rotor cmds) -> RotorGroup -> body wrench
    """

    def __init__(self, cfg: ActionTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]
        self._device = env.device
        self._num_envs = int(env.num_envs)
        self._dt = _get_step_dt(env)

        self._raw_actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self._processed_actions = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)

        params = getattr(cfg, "params", None) or {}
        p_get = params.get if isinstance(params, dict) else lambda k, d=None: getattr(params, k, d)

        self._vel_scale = float(p_get("vel_scale", 1.0))
        self._vel_clip = float(p_get("vel_clip", 6.0))
        self._yaw_rate_scale = float(p_get("yaw_rate_scale", 1.0))
        self._yaw_rate_clip = float(p_get("yaw_rate_clip", self._yaw_rate_scale))
        self._thrust_sign = float(p_get("thrust_sign", 1.0))
        self._g = float(p_get("g", 9.81))
        self._debug_print = bool(p_get("debug_print", False))
        self._debug_interval = max(int(p_get("debug_interval", 100)), 1)
        self._debug_env_id = max(int(p_get("debug_env_id", 0)), 0)
        self._debug_cmd_sat_eps = float(p_get("debug_cmd_sat_eps", 0.995))
        self._debug_counter = 0

        uav_params = p_get("uav_params", None)
        if not isinstance(uav_params, dict):
            uav_params = DRONE_PARAMS
        self._uav_mass = float(uav_params.get("mass", 1.0))

        rotor_cfg = uav_params["rotor_configuration"]
        self._num_rotors = int(rotor_cfg["num_rotors"])

        arm_lengths = torch.as_tensor(rotor_cfg["arm_lengths"], device=self._device, dtype=torch.float32)
        rotor_angles = torch.as_tensor(rotor_cfg["rotor_angles"], device=self._device, dtype=torch.float32)
        self._mix_tau_x = torch.sin(rotor_angles) * arm_lengths
        self._mix_tau_y = -torch.cos(rotor_angles) * arm_lengths

        self._controller = LeePositionController(g=self._g, uav_params=uav_params).to(self._device)
        self._controller.eval()

        self._actuator = RotorGroup(rotor_cfg, dt=self._dt).to(self._device)
        self._actuator.eval()

        self._forces = torch.zeros((self._num_envs, 1, 3), device=self._device, dtype=torch.float32)
        self._torques = torch.zeros((self._num_envs, 1, 3), device=self._device, dtype=torch.float32)
        self._rotor_cmds = torch.zeros((self._num_envs, self._num_rotors), device=self._device, dtype=torch.float32)

        body_id = 0
        try:
            ids, _ = self._asset.find_bodies(
                ["base_link", ".*base.*", "body", ".*body.*"], preserve_order=True
            )
            if len(ids) > 0:
                body_id = int(ids[0])
        except Exception:
            pass
        self._body_ids = [body_id]

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._forces.zero_()
            self._torques.zero_()
            self._rotor_cmds.zero_()
            self._actuator.reset(None)
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._forces[env_ids] = 0.0
            self._torques[env_ids] = 0.0
            self._rotor_cmds[env_ids] = 0.0
            self._actuator.reset(env_ids)

    def process_actions(self, actions: torch.Tensor):
        actions = actions.to(self._device, dtype=torch.float32)
        actions = torch.clamp(actions, -1.0, 1.0)
        self._raw_actions.copy_(actions)

        target_vel = actions[:, 0:3] * self._vel_scale
        if self._vel_clip > 0.0:
            target_vel = torch.clamp(target_vel, -self._vel_clip, self._vel_clip)

        target_yaw_rate = actions[:, 3:4] * self._yaw_rate_scale
        if self._yaw_rate_clip > 0.0:
            target_yaw_rate = torch.clamp(target_yaw_rate, -self._yaw_rate_clip, self._yaw_rate_clip)

        self._processed_actions[:, 0:3] = target_vel
        self._processed_actions[:, 3:4] = target_yaw_rate

    def apply_actions(self):
        root = self._asset.data.root_link_state_w

        target_vel, target_yaw_rate = self._controller.process_rl_actions(self._processed_actions)
        rotor_cmds = self._controller.compute(
            root_state=root,
            target_pos=None,
            target_vel=target_vel,
            target_acc=None,
            target_yaw_rate=target_yaw_rate,
            body_rate=False,
        )
        rotor_cmds = torch.clamp(rotor_cmds, -1.0, 1.0)
        self._rotor_cmds.copy_(rotor_cmds)

        rotor_thrusts, rotor_moments = self._actuator(rotor_cmds)

        total_thrust = rotor_thrusts.sum(dim=-1)
        tau_x = (rotor_thrusts * self._mix_tau_x).sum(dim=-1)
        tau_y = (rotor_thrusts * self._mix_tau_y).sum(dim=-1)
        tau_z = rotor_moments.sum(dim=-1)

        self._forces.zero_()
        self._torques.zero_()
        self._forces[:, 0, 2] = self._thrust_sign * total_thrust
        self._torques[:, 0, 0] = tau_x
        self._torques[:, 0, 1] = tau_y
        self._torques[:, 0, 2] = tau_z

        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=self._forces,
            torques=self._torques,
            positions=None,
            body_ids=self._body_ids,
            env_ids=None,
            is_global=False,
        )
        self._debug_counter += 1
        if self._debug_print and (self._debug_counter % self._debug_interval == 0):
            self._print_debug(
                root=root,
                rotor_cmds=rotor_cmds,
                rotor_thrusts=rotor_thrusts,
                total_thrust=total_thrust,
                tau_x=tau_x,
                tau_y=tau_y,
                tau_z=tau_z,
            )

    # ------------------------------------------------------------------
    # Debug output
    # ------------------------------------------------------------------
    def _print_debug(
        self,
        root: torch.Tensor,
        rotor_cmds: torch.Tensor,
        rotor_thrusts: torch.Tensor,
        total_thrust: torch.Tensor,
        tau_x: torch.Tensor,
        tau_y: torch.Tensor,
        tau_z: torch.Tensor,
    ) -> None:
        try:
            env_id = min(self._debug_env_id, self._num_envs - 1)
            cmd_sat_ratio = (rotor_cmds.abs() >= self._debug_cmd_sat_eps).to(torch.float32).mean(dim=-1)
            thrust_to_weight = total_thrust / max(self._uav_mass * self._g, 1e-6)
            z = root[:, 2]
            vz = root[:, 9]
            print(
                "[CTRL_DEBUG] "
                f"step={self._debug_counter} env={env_id} "
                f"z={z[env_id].item():+.3f} vz={vz[env_id].item():+.3f} "
                f"T={total_thrust[env_id].item():+.3f}N T/W={thrust_to_weight[env_id].item():+.3f} "
                f"tau=({tau_x[env_id].item():+.4f},{tau_y[env_id].item():+.4f},{tau_z[env_id].item():+.4f}) "
                f"cmd[min,max]=({rotor_cmds[env_id].min().item():+.3f},{rotor_cmds[env_id].max().item():+.3f}) "
                f"cmd_sat={cmd_sat_ratio[env_id].item():.2%} "
                f"mean_cmd_sat={cmd_sat_ratio.mean().item():.2%} "
                f"mean_thrust={rotor_thrusts.mean().item():+.3f}",
                flush=True,
            )
        except Exception:
            pass
