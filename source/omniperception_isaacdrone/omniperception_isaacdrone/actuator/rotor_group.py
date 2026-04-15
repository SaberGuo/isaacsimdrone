from __future__ import annotations

import math

import torch
import torch.nn as nn


class RotorGroup(nn.Module):
    """Rotor actuator dynamics ported from OmniPerception/OmniDrones.

    This implementation keeps the same actuator law while adding batched state
    support for vectorized IsaacLab environments.
    """

    def __init__(self, rotor_config: dict, dt: float):
        super().__init__()

        force_constants = torch.as_tensor(rotor_config["force_constants"], dtype=torch.float32)
        moment_constants = torch.as_tensor(rotor_config["moment_constants"], dtype=torch.float32)
        max_rot_vels = torch.as_tensor(rotor_config["max_rotation_velocities"], dtype=torch.float32)

        self.num_rotors = int(len(force_constants))
        self.dt = float(dt)

        self.KF = nn.Parameter(max_rot_vels.square() * force_constants)
        self.KM = nn.Parameter(max_rot_vels.square() * moment_constants)
        self.directions = nn.Parameter(torch.as_tensor(rotor_config["directions"], dtype=torch.float32))

        self.tau_up = nn.Parameter(0.43 * torch.ones(self.num_rotors, dtype=torch.float32))
        self.tau_down = nn.Parameter(0.43 * torch.ones(self.num_rotors, dtype=torch.float32))

        self.noise_scale = 0.002
        self.f = torch.square
        self.f_inv = torch.sqrt

        self.register_buffer("_throttle_state", torch.zeros(0, self.num_rotors, dtype=torch.float32))
        self.requires_grad_(False)

    def _ensure_batch(self, batch_size: int, device: torch.device):
        if self._throttle_state.shape[0] != batch_size or self._throttle_state.device != device:
            self._throttle_state = torch.zeros(batch_size, self.num_rotors, device=device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor | None = None):
        if self._throttle_state.numel() == 0:
            return
        if env_ids is None:
            self._throttle_state.zero_()
        else:
            self._throttle_state[env_ids] = 0.0

    def forward(self, cmds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:
            cmds: (..., num_rotors), normalized rotor commands in [-1, 1].

        Returns:
            thrusts: (..., num_rotors)
            moments: (..., num_rotors), yaw moments around rotor axis.
        """

        if cmds.shape[-1] != self.num_rotors:
            raise ValueError(
                f"Rotor command dim mismatch: got {cmds.shape[-1]}, expected {self.num_rotors}."
            )

        cmds = torch.clamp(cmds.to(torch.float32), -1.0, 1.0)
        leading_shape = cmds.shape[:-1]
        batch_size = int(math.prod(leading_shape)) if len(leading_shape) > 0 else 1

        cmds_flat = cmds.reshape(batch_size, self.num_rotors)
        self._ensure_batch(batch_size, cmds_flat.device)

        target_throttle = self.f_inv(torch.clamp((cmds_flat + 1.0) * 0.5, 0.0, 1.0))

        tau = torch.where(target_throttle > self._throttle_state, self.tau_up, self.tau_down)
        tau = torch.clamp(tau, 0.0, 1.0)
        self._throttle_state.add_(tau * (target_throttle - self._throttle_state))

        noise = torch.randn_like(self._throttle_state) * self.noise_scale * 0.0
        t = torch.clamp(self.f(self._throttle_state) + noise, 0.0, 1.0)

        thrusts = t * self.KF
        moments = (t * self.KM) * -self.directions
        return thrusts.reshape(*leading_shape, self.num_rotors), moments.reshape(*leading_shape, self.num_rotors)
