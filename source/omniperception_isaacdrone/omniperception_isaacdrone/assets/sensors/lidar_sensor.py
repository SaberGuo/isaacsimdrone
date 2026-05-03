"""Project-local LiDAR sensor wrapper for body-frame point clouds."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.sensors import LidarSensor
from isaaclab.sensors.ray_caster import RayCaster


# -----------------------------------------------------------------------------
# LiDAR sensor
# -----------------------------------------------------------------------------
class BodyFrameLidarSensor(LidarSensor):
    """Project-local LiDAR wrapper for body-frame scans and stable Livox timing.

    Two project-specific behaviors are enforced here:
    1. The sensor is refreshed every physics step so observations/rewards always read a fresh point cloud.
    2. The dynamic Livox-like scan phase advances on the configured ``update_frequency`` cadence
       instead of incorrectly depending on ``cfg.update_period``.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._last_update_dt = float(getattr(self, "_sim_physics_dt", 0.0)) or (1.0 / 60.0)
        self._dynamic_elapsed = 0.0
        self._dynamic_phase_time = 0.0
        self._base_ray_directions: torch.Tensor | None = None

    def update(self, dt: float, force_recompute: bool = False):
        # The RL environment reads LiDAR every physics step, so keep the sensor outdated every step.
        self._last_update_dt = float(dt)
        self._timestamp += dt
        self._is_outdated[:] = True
        if force_recompute or self._is_visualizing or (self.cfg.history_length > 0):
            self._update_outdated_buffers()

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        self._base_ray_directions = self.ray_directions.clone()

    def _uses_dynamic_pattern(self) -> bool:
        pattern_cfg = getattr(self.cfg, "pattern_cfg", None)
        if pattern_cfg is None:
            return False
        if not hasattr(pattern_cfg, "sensor_type"):
            return False
        return not bool(getattr(pattern_cfg, "use_simple_grid", True))

    def _advance_dynamic_pattern(self, dt: float) -> None:
        if not self._uses_dynamic_pattern():
            return

        self._dynamic_elapsed += max(float(dt), 0.0)
        phase_advanced = False
        while self._dynamic_elapsed + 1.0e-6 >= self.update_dt:
            self._dynamic_elapsed -= self.update_dt
            self._dynamic_phase_time += self.update_dt
            phase_advanced = True

        if phase_advanced:
            self.sensor_t = self._dynamic_phase_time
            self._update_dynamic_rays()

    def _update_dynamic_rays(self):
        if self._base_ray_directions is None:
            self._base_ray_directions = self.ray_directions.clone()

        rotation_angle = self._dynamic_phase_time * 0.1
        cos_rot = math.cos(rotation_angle)
        sin_rot = math.sin(rotation_angle)

        base_dirs = self._base_ray_directions
        self.ray_directions[..., 0] = base_dirs[..., 0] * cos_rot - base_dirs[..., 1] * sin_rot
        self.ray_directions[..., 1] = base_dirs[..., 0] * sin_rot + base_dirs[..., 1] * cos_rot
        self.ray_directions[..., 2] = base_dirs[..., 2]

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        self._advance_dynamic_pattern(self._last_update_dt)

        # Bypass the IsaacLab LiDAR timing path and reuse only the ray-cast + post-processing.
        RayCaster._update_buffers_impl(self, env_ids)

        sensor_pos = self._get_true_sensor_pos()[env_ids].unsqueeze(1)
        hit_points = self._data.ray_hits_w[env_ids]
        distances = torch.norm(hit_points - sensor_pos, dim=2)

        inf_mask = torch.isinf(hit_points).any(dim=2)
        distances[inf_mask] = self.cfg.max_distance

        if self.cfg.enable_sensor_noise:
            distances = self._apply_noise(distances, env_ids)

        self._data.distances[env_ids] = distances

        if self.cfg.return_pointcloud:
            self._generate_pointcloud(env_ids)
