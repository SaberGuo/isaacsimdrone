from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def quat_to_rotmat_wxyz(quat_wxyz: Iterable[float]) -> np.ndarray:
    q = np.asarray(list(quat_wxyz), dtype=np.float64).reshape(4)
    w, x, y, z = q
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    return np.array(
        [
            [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def quat_to_yaw(quat_wxyz: Iterable[float]) -> float:
    r = quat_to_rotmat_wxyz(quat_wxyz)
    return float(math.atan2(r[1, 0], r[0, 0]))


def yaw_rotation_matrix(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def body_points_to_yaw_frame(points_body: np.ndarray, quat_wxyz: Iterable[float]) -> np.ndarray:
    pts = np.asarray(points_body, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    r_body = quat_to_rotmat_wxyz(quat_wxyz)
    r_yaw = yaw_rotation_matrix(quat_to_yaw(quat_wxyz))
    yaw_from_body = r_yaw.T @ r_body
    return pts @ yaw_from_body.T


def world_points_to_yaw_frame(
    points_world: np.ndarray,
    origin_world: Iterable[float],
    quat_wxyz: Iterable[float],
) -> np.ndarray:
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    origin = np.asarray(list(origin_world), dtype=np.float64).reshape(3)
    r_yaw = yaw_rotation_matrix(quat_to_yaw(quat_wxyz))
    rel = pts - origin.reshape(1, 3)
    return rel @ r_yaw


def yaw_frame_points_to_world(
    points_local: np.ndarray,
    origin_world: Iterable[float],
    quat_wxyz: Iterable[float],
) -> np.ndarray:
    pts = np.asarray(points_local, dtype=np.float64)
    if pts.size == 0:
        return pts.reshape(0, 3)
    origin = np.asarray(list(origin_world), dtype=np.float64).reshape(3)
    r_yaw = yaw_rotation_matrix(quat_to_yaw(quat_wxyz))
    return pts @ r_yaw.T + origin.reshape(1, 3)


@dataclass
class OccupancyStats:
    occupied_voxels: int
    inflated_voxels: int
    occupied_ratio: float
    num_points_used: int


class LocalOccupancyMap:
    """Local yaw-aligned occupancy grid built from LiDAR pointcloud."""

    def __init__(
        self,
        resolution: float = 0.4,
        x_range: tuple[float, float] = (-12.0, 12.0),
        y_range: tuple[float, float] = (-12.0, 12.0),
        z_range: tuple[float, float] = (-4.0, 4.0),
        inflation_radius: float = 1.4,
        min_distance_from_robot: float = 0.25,
    ) -> None:
        self.resolution = float(resolution)
        self.x_range = (float(x_range[0]), float(x_range[1]))
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self.z_range = (float(z_range[0]), float(z_range[1]))
        self.min_distance_from_robot = float(min_distance_from_robot)

        self.min_bound = np.array(
            [self.x_range[0], self.y_range[0], self.z_range[0]],
            dtype=np.float64,
        )
        self.max_bound = np.array(
            [self.x_range[1], self.y_range[1], self.z_range[1]],
            dtype=np.float64,
        )
        dims = np.ceil((self.max_bound - self.min_bound) / self.resolution).astype(np.int32)
        self.grid_shape = tuple(int(v) for v in dims.tolist())
        self.occupied = np.zeros(self.grid_shape, dtype=bool)
        self.inflated = np.zeros(self.grid_shape, dtype=bool)

        self.inflation_radius = float(inflation_radius)
        inflate_steps = max(int(math.ceil(self.inflation_radius / self.resolution)), 0)
        self.inflate_offsets = self._build_offsets(inflate_steps)
        self.stats = OccupancyStats(0, 0, 0.0, 0)

    def _build_offsets(self, inflate_steps: int) -> np.ndarray:
        offsets: list[tuple[int, int, int]] = []
        for dx in range(-inflate_steps, inflate_steps + 1):
            for dy in range(-inflate_steps, inflate_steps + 1):
                for dz in range(-inflate_steps, inflate_steps + 1):
                    if dx * dx + dy * dy + dz * dz <= inflate_steps * inflate_steps:
                        offsets.append((dx, dy, dz))
        return np.asarray(offsets, dtype=np.int32)

    def clear(self) -> None:
        self.occupied.fill(False)
        self.inflated.fill(False)
        self.stats = OccupancyStats(0, 0, 0.0, 0)

    def point_to_index(self, point_local: Iterable[float]) -> tuple[int, int, int] | None:
        p = np.asarray(list(point_local), dtype=np.float64).reshape(3)
        if np.any(p < self.min_bound) or np.any(p >= self.max_bound):
            return None
        idx = np.floor((p - self.min_bound) / self.resolution).astype(np.int32)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def index_to_point(self, index: Iterable[int]) -> np.ndarray:
        idx = np.asarray(list(index), dtype=np.float64).reshape(3)
        return self.min_bound + (idx + 0.5) * self.resolution

    def clamp_point(self, point_local: Iterable[float], margin: float = 0.2) -> np.ndarray:
        margin_vec = np.full((3,), max(float(margin), 0.0), dtype=np.float64)
        return np.clip(
            np.asarray(list(point_local), dtype=np.float64).reshape(3),
            self.min_bound + margin_vec,
            self.max_bound - margin_vec,
        )

    def is_in_bounds(self, point_local: Iterable[float]) -> bool:
        return self.point_to_index(point_local) is not None

    def is_occupied_index(self, index: Iterable[int]) -> bool:
        idx = np.asarray(list(index), dtype=np.int32).reshape(3)
        if np.any(idx < 0) or np.any(idx >= np.asarray(self.grid_shape, dtype=np.int32)):
            return True
        return bool(self.inflated[idx[0], idx[1], idx[2]])

    def is_occupied_point(self, point_local: Iterable[float]) -> bool:
        idx = self.point_to_index(point_local)
        if idx is None:
            return True
        return self.is_occupied_index(idx)

    def _inflate(self) -> None:
        self.inflated.fill(False)
        occ_idx = np.argwhere(self.occupied)
        if occ_idx.size == 0:
            return
        self.inflated[self.occupied] = True
        for dx, dy, dz in self.inflate_offsets:
            src_x0 = max(0, -dx)
            src_x1 = min(self.grid_shape[0], self.grid_shape[0] - dx)
            src_y0 = max(0, -dy)
            src_y1 = min(self.grid_shape[1], self.grid_shape[1] - dy)
            src_z0 = max(0, -dz)
            src_z1 = min(self.grid_shape[2], self.grid_shape[2] - dz)
            dst_x0 = max(0, dx)
            dst_x1 = min(self.grid_shape[0], self.grid_shape[0] + dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(self.grid_shape[1], self.grid_shape[1] + dy)
            dst_z0 = max(0, dz)
            dst_z1 = min(self.grid_shape[2], self.grid_shape[2] + dz)
            self.inflated[dst_x0:dst_x1, dst_y0:dst_y1, dst_z0:dst_z1] |= self.occupied[
                src_x0:src_x1,
                src_y0:src_y1,
                src_z0:src_z1,
            ]

    def update_from_pointcloud_body(
        self,
        points_body: np.ndarray,
        quat_wxyz: Iterable[float],
    ) -> OccupancyStats:
        self.clear()
        points_local = body_points_to_yaw_frame(points_body, quat_wxyz)
        if points_local.size == 0:
            return self.stats

        finite_mask = np.isfinite(points_local).all(axis=1)
        dist_xy = np.linalg.norm(points_local[:, :2], axis=1)
        valid = finite_mask & (dist_xy >= self.min_distance_from_robot)
        valid &= np.all(points_local >= self.min_bound.reshape(1, 3), axis=1)
        valid &= np.all(points_local < self.max_bound.reshape(1, 3), axis=1)
        valid_points = points_local[valid]
        if valid_points.size == 0:
            return self.stats

        idx = np.floor((valid_points - self.min_bound.reshape(1, 3)) / self.resolution).astype(np.int32)
        idx = np.unique(idx, axis=0)
        self.occupied[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        self._inflate()

        occupied_voxels = int(self.occupied.sum())
        inflated_voxels = int(self.inflated.sum())
        ratio = inflated_voxels / float(np.prod(self.grid_shape))
        self.stats = OccupancyStats(
            occupied_voxels=occupied_voxels,
            inflated_voxels=inflated_voxels,
            occupied_ratio=float(ratio),
            num_points_used=int(len(valid_points)),
        )
        return self.stats

    def collision_along_local_points(self, points_local: np.ndarray) -> bool:
        pts = np.asarray(points_local, dtype=np.float64).reshape(-1, 3)
        if pts.size == 0:
            return False
        for p in pts:
            if self.is_occupied_point(p):
                return True
        return False

    def collision_along_world_points(
        self,
        points_world: np.ndarray,
        origin_world: Iterable[float],
        quat_wxyz: Iterable[float],
    ) -> bool:
        local = world_points_to_yaw_frame(points_world, origin_world, quat_wxyz)
        return self.collision_along_local_points(local)

    def shortcut_is_free(
        self,
        p0_local: Iterable[float],
        p1_local: Iterable[float],
        sample_step: float | None = None,
    ) -> bool:
        p0 = np.asarray(list(p0_local), dtype=np.float64).reshape(3)
        p1 = np.asarray(list(p1_local), dtype=np.float64).reshape(3)
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1.0e-6:
            return not self.is_occupied_point(p0)
        step = float(sample_step) if sample_step is not None else self.resolution * 0.5
        num = max(int(math.ceil(length / max(step, 1.0e-3))), 2)
        alphas = np.linspace(0.0, 1.0, num=num, dtype=np.float64)
        samples = p0.reshape(1, 3) + alphas.reshape(-1, 1) * seg.reshape(1, 3)
        return not self.collision_along_local_points(samples)
