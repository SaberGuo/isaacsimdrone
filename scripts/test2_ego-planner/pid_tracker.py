from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from occupancy_map import wrap_to_pi


@dataclass
class PIDConfig:
    kp_xyz: tuple[float, float, float] = (0.55, 0.55, 0.68)
    ki_xyz: tuple[float, float, float] = (0.00, 0.00, 0.00)
    kd_xyz: tuple[float, float, float] = (0.16, 0.16, 0.20)
    integral_limit_xyz: tuple[float, float, float] = (0.8, 0.8, 0.5)
    integral_leak: float = 0.92
    pos_error_clip_xyz: tuple[float, float, float] = (4.5, 4.5, 1.0)
    feedforward_vel_gain: float = 0.92
    feedforward_acc_gain: float = 0.00
    max_speed_xy: float = 3.6
    max_speed_z: float = 0.90
    max_accel_xy: float = 6.0
    max_accel_z: float = 2.0
    max_decel_xy: float = 8.0
    max_decel_z: float = 3.5
    cmd_time_constant: float = 0.08
    brake_time_constant: float = 0.03
    action_vel_scale: float = 1.0
    yaw_deadband_speed: float = 0.20
    yaw_slew_rate: float = 1.35
    yaw_track_blend: float = 0.18


class PositionPIDTracker:
    """Outer-loop path follower for the test6 velocity+yaw action interface."""

    def __init__(self, cfg: PIDConfig) -> None:
        self.cfg = cfg
        self.kp = np.asarray(cfg.kp_xyz, dtype=np.float64)
        self.ki = np.asarray(cfg.ki_xyz, dtype=np.float64)
        self.kd = np.asarray(cfg.kd_xyz, dtype=np.float64)
        self.integral_limit = np.asarray(cfg.integral_limit_xyz, dtype=np.float64)
        self.pos_error_clip = np.asarray(cfg.pos_error_clip_xyz, dtype=np.float64)
        self.integral_error = np.zeros(3, dtype=np.float64)
        self.filtered_velocity_cmd = np.zeros(3, dtype=np.float64)
        self.yaw_target = 0.0
        self.is_initialized = False

    def reset(self) -> None:
        self.integral_error.fill(0.0)
        self.filtered_velocity_cmd.fill(0.0)
        self.yaw_target = 0.0
        self.is_initialized = False

    def _clip_velocity(self, vel_world: np.ndarray, speed_scale: float = 1.0) -> np.ndarray:
        vel = vel_world.copy()
        speed_scale = float(np.clip(speed_scale, 0.0, 1.0))
        max_speed_xy = self.cfg.max_speed_xy * speed_scale
        max_speed_z = self.cfg.max_speed_z * speed_scale
        xy_norm = float(np.linalg.norm(vel[:2]))
        if xy_norm > max_speed_xy > 0.0:
            vel[:2] *= max_speed_xy / max(xy_norm, 1.0e-6)
        elif max_speed_xy <= 0.0:
            vel[:2] = 0.0
        vel[2] = float(np.clip(vel[2], -max_speed_z, max_speed_z))
        return vel

    def _limit_accel(self, target_vel: np.ndarray, dt: float, emergency_brake: bool = False) -> np.ndarray:
        delta = np.asarray(target_vel, dtype=np.float64) - self.filtered_velocity_cmd
        current_xy = float(np.linalg.norm(self.filtered_velocity_cmd[:2]))
        target_xy = float(np.linalg.norm(target_vel[:2]))
        braking_xy = emergency_brake or target_xy + 1.0e-3 < current_xy or np.dot(
            target_vel[:2], self.filtered_velocity_cmd[:2]
        ) < 0.0
        delta_xy_norm = float(np.linalg.norm(delta[:2]))
        max_delta_xy = (self.cfg.max_decel_xy if braking_xy else self.cfg.max_accel_xy) * dt
        if delta_xy_norm > max_delta_xy > 0.0:
            delta[:2] *= max_delta_xy / max(delta_xy_norm, 1.0e-6)
        current_z = float(abs(self.filtered_velocity_cmd[2]))
        target_z = float(abs(target_vel[2]))
        braking_z = emergency_brake or target_z + 1.0e-3 < current_z or (
            target_vel[2] * self.filtered_velocity_cmd[2]
        ) < 0.0
        max_delta_z = (self.cfg.max_decel_z if braking_z else self.cfg.max_accel_z) * dt
        delta[2] = float(np.clip(delta[2], -max_delta_z, max_delta_z))
        return self.filtered_velocity_cmd + delta

    def _smooth_velocity(self, target_vel: np.ndarray, dt: float, emergency_brake: bool = False) -> np.ndarray:
        current_speed = float(np.linalg.norm(self.filtered_velocity_cmd))
        target_speed = float(np.linalg.norm(target_vel))
        braking = emergency_brake or target_speed + 1.0e-3 < current_speed or np.dot(
            target_vel, self.filtered_velocity_cmd
        ) < 0.0
        tau = self.cfg.brake_time_constant if braking else self.cfg.cmd_time_constant
        tau = max(tau, 1.0e-3)
        alpha = np.clip(dt / (tau + dt), 0.0, 1.0)
        return (1.0 - alpha) * self.filtered_velocity_cmd + alpha * np.asarray(target_vel, dtype=np.float64)

    def _update_yaw_target(self, current_yaw: float, velocity_cmd: np.ndarray, track_dir: np.ndarray, dt: float) -> float:
        planar_cmd = np.asarray(velocity_cmd[:2], dtype=np.float64)
        planar_track = np.asarray(track_dir[:2], dtype=np.float64)
        speed = float(np.linalg.norm(planar_cmd))

        desired_yaw = self.yaw_target if self.is_initialized else float(current_yaw)
        if np.linalg.norm(planar_track) >= self.cfg.yaw_deadband_speed:
            yaw_track = float(math.atan2(planar_track[1], planar_track[0]))
            if np.linalg.norm(planar_cmd) >= self.cfg.yaw_deadband_speed:
                yaw_cmd = float(math.atan2(planar_cmd[1], planar_cmd[0]))
                yaw_mix = wrap_to_pi(yaw_cmd - yaw_track) * self.cfg.yaw_track_blend
                desired_yaw = wrap_to_pi(yaw_track + yaw_mix)
            else:
                desired_yaw = yaw_track
        elif speed >= self.cfg.yaw_deadband_speed:
            desired_yaw = float(math.atan2(planar_cmd[1], planar_cmd[0]))
        else:
            desired_yaw = self.yaw_target if self.is_initialized else float(current_yaw)

        yaw_prev = self.yaw_target if self.is_initialized else float(current_yaw)
        max_delta = self.cfg.yaw_slew_rate * dt
        yaw_delta = wrap_to_pi(desired_yaw - yaw_prev)
        yaw_delta = float(np.clip(yaw_delta, -max_delta, max_delta))
        self.yaw_target = wrap_to_pi(yaw_prev + yaw_delta)
        return self.yaw_target

    def compute_action(
        self,
        dt: float,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_yaw: float,
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
        ref_acc: np.ndarray,
        max_speed_scale: float = 1.0,
        emergency_brake: bool = False,
    ) -> tuple[np.ndarray, dict[str, float]]:
        dt = max(float(dt), 1.0e-3)
        current_pos = np.asarray(current_pos, dtype=np.float64)
        current_vel = np.asarray(current_vel, dtype=np.float64)
        ref_pos = np.asarray(ref_pos, dtype=np.float64)
        ref_vel = np.asarray(ref_vel, dtype=np.float64)
        ref_acc = np.asarray(ref_acc, dtype=np.float64)

        pos_error = np.clip(ref_pos - current_pos, -self.pos_error_clip, self.pos_error_clip)
        vel_error = ref_vel - current_vel

        self.integral_error *= self.cfg.integral_leak
        self.integral_error += pos_error * dt
        self.integral_error = np.clip(self.integral_error, -self.integral_limit, self.integral_limit)

        target_vel = (
            self.cfg.feedforward_vel_gain * ref_vel
            + self.kp * pos_error
            + self.ki * self.integral_error
            + self.kd * vel_error
            + self.cfg.feedforward_acc_gain * ref_acc
        )
        target_vel = self._clip_velocity(target_vel, speed_scale=max_speed_scale)

        if not self.is_initialized:
            self.filtered_velocity_cmd = target_vel.copy()
            self.yaw_target = float(current_yaw)
            self.is_initialized = True
        else:
            accel_limited = self._limit_accel(target_vel, dt, emergency_brake=emergency_brake)
            self.filtered_velocity_cmd = self._smooth_velocity(
                accel_limited, dt, emergency_brake=emergency_brake
            )
            self.filtered_velocity_cmd = self._clip_velocity(
                self.filtered_velocity_cmd, speed_scale=max_speed_scale
            )

        desired_yaw = self._update_yaw_target(
            current_yaw=float(current_yaw),
            velocity_cmd=self.filtered_velocity_cmd,
            track_dir=ref_vel,
            dt=dt,
        )

        action = np.zeros(4, dtype=np.float32)
        action[:3] = np.clip(
            self.filtered_velocity_cmd / max(self.cfg.action_vel_scale, 1.0e-6),
            -1.0,
            1.0,
        )
        action[3] = np.clip(wrap_to_pi(desired_yaw) / math.pi, -1.0, 1.0)

        debug = {
            "pos_error_norm": float(np.linalg.norm(pos_error)),
            "vel_error_norm": float(np.linalg.norm(vel_error)),
            "target_speed": float(np.linalg.norm(target_vel)),
            "command_speed": float(np.linalg.norm(self.filtered_velocity_cmd)),
            "speed_scale": float(max_speed_scale),
            "emergency_brake": float(bool(emergency_brake)),
            "desired_yaw_deg": math.degrees(desired_yaw),
            "xy_cmd_norm": float(np.linalg.norm(self.filtered_velocity_cmd[:2])),
            "z_cmd": float(self.filtered_velocity_cmd[2]),
        }
        return action, debug
