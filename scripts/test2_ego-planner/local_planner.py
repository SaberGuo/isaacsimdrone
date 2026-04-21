from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ego_math import (
    PolynomialTrajectory,
    TimedPolyline,
    TrajectorySample,
    UniformBSpline,
    densify_polyline,
    ensure_min_points,
)
from occupancy_map import LocalOccupancyMap, world_points_to_yaw_frame, yaw_frame_points_to_world


def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


@dataclass
class PlannerConfig:
    max_vel: float = 4.5
    max_acc: float = 6.5
    feasibility_tolerance: float = 0.05
    ctrl_pt_dist: float = 0.9
    planning_horizon: float = 10.0
    global_segment_max_dist: float = 4.0
    local_target_margin: float = 0.4
    min_plan_points: int = 7
    min_plan_duration_s: float = 0.6
    collision_check_dt: float = 0.05
    replan_period_s: float = 0.9
    min_remaining_s: float = 0.7
    astar_max_expansions: int = 50000
    goal_tolerance: float = 0.6
    vertical_cost_weight: float = 4.0
    level_flight_z_limit: float = 0.35
    projection_window_back_s: float = 0.4
    projection_window_forward_s: float = 2.0
    projection_dt_s: float = 0.05
    tracking_lookahead_s: float = 0.85


@dataclass
class GlobalReference:
    trajectory: PolynomialTrajectory
    start_world: np.ndarray
    goal_world: np.ndarray
    duration: float
    last_progress_time: float = 0.0


@dataclass
class LocalPlan:
    trajectory: Any
    start_time: float
    duration: float
    goal_world: np.ndarray
    local_target_world: np.ndarray
    path_world: np.ndarray
    path_local: np.ndarray
    mode: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def sample(self, sim_time: float) -> TrajectorySample:
        t = max(0.0, min(float(sim_time - self.start_time), self.duration))
        return self.trajectory.sample(t)

    def sample_relative(self, rel_t: float) -> TrajectorySample:
        t = max(0.0, min(float(rel_t), self.duration))
        return self.trajectory.sample(t)

    def remaining_time(self, sim_time: float) -> float:
        return max(0.0, float(self.duration - (sim_time - self.start_time)))

    def project_progress_time(
        self,
        sim_time: float,
        current_pos: np.ndarray,
        window_back_s: float,
        window_forward_s: float,
        sample_dt_s: float,
    ) -> float:
        age = float(np.clip(sim_time - self.start_time, 0.0, self.duration))
        t_min = max(0.0, age - float(window_back_s))
        t_max = min(self.duration, age + float(window_forward_s))
        if t_max <= t_min + 1.0e-6:
            return age

        ts = np.arange(t_min, t_max + 1.0e-6, max(float(sample_dt_s), 1.0e-3), dtype=np.float64)
        if ts.size == 0:
            ts = np.array([age], dtype=np.float64)
        positions = np.stack([self.trajectory.sample(t).position for t in ts], axis=0)
        dist = np.linalg.norm(positions - np.asarray(current_pos, dtype=np.float64).reshape(1, 3), axis=1)
        best_idx = int(np.argmin(dist))
        return float(ts[best_idx])

    def sample_tracking_reference(
        self,
        sim_time: float,
        current_pos: np.ndarray,
        lookahead_s: float,
        window_back_s: float,
        window_forward_s: float,
        sample_dt_s: float,
    ) -> tuple[TrajectorySample, float, float]:
        progress_t = self.project_progress_time(
            sim_time=sim_time,
            current_pos=current_pos,
            window_back_s=window_back_s,
            window_forward_s=window_forward_s,
            sample_dt_s=sample_dt_s,
        )
        target_t = min(self.duration, progress_t + max(float(lookahead_s), 0.0))
        return self.sample_relative(target_t), progress_t, target_t


class GridAStar:
    def __init__(
        self,
        occupancy_map: LocalOccupancyMap,
        max_expansions: int = 50000,
        vertical_cost_weight: float = 4.0,
    ) -> None:
        self.map = occupancy_map
        self.max_expansions = int(max_expansions)
        self.vertical_cost_weight = max(float(vertical_cost_weight), 1.0)
        self.neighbors = [
            (
                dx,
                dy,
                dz,
                math.sqrt(dx * dx + dy * dy + (self.vertical_cost_weight * dz) * (self.vertical_cost_weight * dz)),
            )
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

    def _heuristic(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        diff[2] *= self.vertical_cost_weight
        return 1.0001 * float(np.linalg.norm(diff))

    def _nearest_free(self, idx: tuple[int, int, int], max_radius: int = 4) -> tuple[int, int, int] | None:
        if not self.map.is_occupied_index(idx):
            return idx
        base = np.asarray(idx, dtype=np.int32)
        for radius in range(1, max_radius + 1):
            candidates: list[tuple[float, tuple[int, int, int]]] = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dz in range(-radius, radius + 1):
                        cand = base + np.array([dx, dy, dz], dtype=np.int32)
                        cand_t = (int(cand[0]), int(cand[1]), int(cand[2]))
                        if self.map.is_occupied_index(cand_t):
                            continue
                        candidates.append((float(np.linalg.norm(cand - base)), cand_t))
            if candidates:
                candidates.sort(key=lambda item: item[0])
                return candidates[0][1]
        return None

    def search(self, start_local: np.ndarray, goal_local: np.ndarray) -> np.ndarray | None:
        start_idx = self.map.point_to_index(start_local)
        goal_idx = self.map.point_to_index(goal_local)
        if start_idx is None or goal_idx is None:
            return None

        start_idx = self._nearest_free(start_idx)
        goal_idx = self._nearest_free(goal_idx)
        if start_idx is None or goal_idx is None:
            return None

        open_heap: list[tuple[float, int, tuple[int, int, int]]] = []
        counter = 0
        g_score: dict[tuple[int, int, int], float] = {start_idx: 0.0}
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        heapq.heappush(open_heap, (self._heuristic(start_idx, goal_idx), counter, start_idx))
        closed: set[tuple[int, int, int]] = set()

        expansions = 0
        while open_heap and expansions < self.max_expansions:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            expansions += 1
            if current == goal_idx:
                return self._reconstruct_path(came_from, current)
            closed.add(current)

            for dx, dy, dz, step_cost in self.neighbors:
                neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
                if self.map.is_occupied_index(neighbor):
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                counter += 1
                f_score = tentative_g + self._heuristic(neighbor, goal_idx)
                heapq.heappush(open_heap, (f_score, counter, neighbor))
        return None

    def _reconstruct_path(
        self,
        came_from: dict[tuple[int, int, int], tuple[int, int, int]],
        current: tuple[int, int, int],
    ) -> np.ndarray:
        path_idx = [current]
        while current in came_from:
            current = came_from[current]
            path_idx.append(current)
        path_idx.reverse()
        points = np.stack([self.map.index_to_point(idx) for idx in path_idx], axis=0)
        return points


class EgoStyleLocalPlanner:
    def __init__(self, cfg: PlannerConfig, occupancy_map: LocalOccupancyMap) -> None:
        self.cfg = cfg
        self.map = occupancy_map
        self.astar = GridAStar(
            occupancy_map,
            max_expansions=cfg.astar_max_expansions,
            vertical_cost_weight=cfg.vertical_cost_weight,
        )
        self.global_ref: GlobalReference | None = None

    def reset(self) -> None:
        self.global_ref = None

    def goal_reached(self, current_pos: np.ndarray, goal_world: np.ndarray) -> bool:
        return _norm(np.asarray(goal_world, dtype=np.float64) - np.asarray(current_pos, dtype=np.float64)) <= self.cfg.goal_tolerance

    def _build_global_reference(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        start_acc: np.ndarray,
        goal_pos: np.ndarray,
    ) -> GlobalReference:
        points: list[np.ndarray] = [start_pos.copy()]
        delta = goal_pos - start_pos
        distance = _norm(delta)
        if distance > self.cfg.global_segment_max_dist:
            num_segments = int(math.floor(distance / self.cfg.global_segment_max_dist)) + 1
            for i in range(1, num_segments):
                alpha = i / float(num_segments)
                points.append((1.0 - alpha) * start_pos + alpha * goal_pos)
        points.append(goal_pos.copy())

        pos = np.stack(points, axis=1)
        seg_vec = pos[:, 1:] - pos[:, :-1]
        seg_len = np.linalg.norm(seg_vec, axis=0)
        time_alloc = np.maximum(seg_len / max(self.cfg.max_vel, 1.0e-3), 0.25)
        time_alloc[0] *= 2.0
        time_alloc[-1] *= 2.0

        if pos.shape[1] >= 3:
            traj = PolynomialTrajectory.min_snap_traj(
                pos,
                start_vel,
                np.zeros(3, dtype=np.float64),
                start_acc,
                np.zeros(3, dtype=np.float64),
                time_alloc,
            )
        else:
            traj = PolynomialTrajectory.one_segment_traj_gen(
                start_pos,
                start_vel,
                start_acc,
                goal_pos,
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                float(time_alloc[0]),
            )

        return GlobalReference(
            trajectory=traj,
            start_world=start_pos.copy(),
            goal_world=goal_pos.copy(),
            duration=traj.get_time_sum(),
            last_progress_time=0.0,
        )

    def _ensure_global_reference(
        self,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_acc: np.ndarray,
        goal_world: np.ndarray,
    ) -> None:
        if self.global_ref is None:
            self.global_ref = self._build_global_reference(current_pos, current_vel, current_acc, goal_world)
            return
        if _norm(goal_world - self.global_ref.goal_world) > 1.0e-3:
            self.global_ref = self._build_global_reference(current_pos, current_vel, current_acc, goal_world)

    def _get_local_target(
        self,
        current_pos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.global_ref is None:
            raise RuntimeError("Global reference is not initialized.")

        traj = self.global_ref.trajectory
        t = self.global_ref.last_progress_time
        t_step = self.cfg.planning_horizon / 20.0 / max(self.cfg.max_vel, 1.0e-3)
        t_step = max(t_step, 0.05)

        dist_min = 1.0e9
        dist_min_t = self.global_ref.last_progress_time
        local_target = self.global_ref.goal_world.copy()
        local_vel = np.zeros(3, dtype=np.float64)

        while t < self.global_ref.duration + 1.0e-6:
            pos_t = traj.evaluate(t)
            dist = _norm(pos_t - current_pos)

            if t < self.global_ref.last_progress_time + 1.0e-5 and dist > self.cfg.planning_horizon:
                t_probe = t
                while t_probe < self.global_ref.duration + 1.0e-6:
                    pos_probe = traj.evaluate(t_probe)
                    dist_probe = _norm(pos_probe - current_pos)
                    if dist_probe < self.cfg.planning_horizon:
                        pos_t = pos_probe
                        dist = dist_probe
                        t = t_probe
                        break
                    t_probe += t_step

            if dist < dist_min:
                dist_min = dist
                dist_min_t = t

            if dist >= self.cfg.planning_horizon:
                local_target = pos_t
                self.global_ref.last_progress_time = dist_min_t
                if _norm(self.global_ref.goal_world - local_target) < (self.cfg.max_vel ** 2) / (2.0 * max(self.cfg.max_acc, 1.0e-3)):
                    local_vel = np.zeros(3, dtype=np.float64)
                else:
                    local_vel = traj.evaluate_vel(t)
                return local_target, local_vel
            t += t_step

        self.global_ref.last_progress_time = self.global_ref.duration
        return self.global_ref.goal_world.copy(), np.zeros(3, dtype=np.float64)

    def _shortcut_path(self, path_local: np.ndarray) -> np.ndarray:
        if len(path_local) <= 2:
            return path_local.copy()
        simplified: list[np.ndarray] = [path_local[0].copy()]
        i = 0
        while i < len(path_local) - 1:
            j = len(path_local) - 1
            while j > i + 1:
                if self.map.shortcut_is_free(path_local[i], path_local[j]):
                    break
                j -= 1
            simplified.append(path_local[j].copy())
            i = j
        return np.asarray(simplified, dtype=np.float64)

    def _trajectory_collides(
        self,
        trajectory: Any,
        origin_world: np.ndarray,
        current_quat: np.ndarray,
        from_time: float = 0.0,
    ) -> bool:
        duration = float(trajectory.get_time_sum())
        sample_times = np.arange(max(from_time, 0.0), duration + 1.0e-6, self.cfg.collision_check_dt, dtype=np.float64)
        if sample_times.size == 0:
            sample_times = np.array([duration], dtype=np.float64)
        points_world = np.stack([trajectory.sample(t).position for t in sample_times], axis=0)
        return self.map.collision_along_world_points(points_world, origin_world, current_quat)

    def plan(
        self,
        sim_time: float,
        current_pos: np.ndarray,
        current_vel: np.ndarray,
        current_acc: np.ndarray,
        current_quat: np.ndarray,
        goal_world: np.ndarray,
        reason: str,
    ) -> LocalPlan | None:
        self._ensure_global_reference(current_pos, current_vel, current_acc, goal_world)
        local_target_world, local_target_vel = self._get_local_target(current_pos)
        local_target_local = world_points_to_yaw_frame(
            local_target_world.reshape(1, 3),
            current_pos,
            current_quat,
        )[0]
        z_goal_delta = float(np.clip(goal_world[2] - current_pos[2], -self.cfg.level_flight_z_limit, self.cfg.level_flight_z_limit))
        local_target_local[2] = z_goal_delta
        local_target_local = self.map.clamp_point(local_target_local, margin=self.cfg.local_target_margin)

        start_local = np.zeros(3, dtype=np.float64)
        if _norm(local_target_local - start_local) < 0.25:
            local_target_world = current_pos.copy()
            local_target_vel = np.zeros(3, dtype=np.float64)
            trajectory = TimedPolyline(np.stack([current_pos, current_pos], axis=0), max_vel=max(self.cfg.max_vel, 1.0))
            return LocalPlan(
                trajectory=trajectory,
                start_time=float(sim_time),
                duration=max(self.cfg.min_plan_duration_s, trajectory.get_time_sum()),
                goal_world=goal_world.copy(),
                local_target_world=local_target_world.copy(),
                path_world=np.stack([current_pos, current_pos], axis=0),
                path_local=np.stack([start_local, start_local], axis=0),
                mode="hover",
                metadata={"reason": reason, "local_target_clamped": True},
            )

        if self.map.shortcut_is_free(start_local, local_target_local):
            path_local = np.stack([start_local, local_target_local], axis=0)
        else:
            path_local = self.astar.search(start_local, local_target_local)
            if path_local is None or len(path_local) < 2:
                return None

        path_local = self._shortcut_path(path_local)
        if path_local.shape[0] >= 2:
            path_local[:, 2] = np.linspace(0.0, z_goal_delta, path_local.shape[0], dtype=np.float64)
        path_world = yaw_frame_points_to_world(path_local, current_pos, current_quat)
        path_world = densify_polyline(path_world, spacing=self.cfg.ctrl_pt_dist)
        path_world = ensure_min_points(path_world, self.cfg.min_plan_points)

        ts = max(self.cfg.ctrl_pt_dist / max(self.cfg.max_vel, 1.0e-3) * 1.5, 0.08)
        start_end = [
            current_vel.copy(),
            local_target_vel.copy(),
            current_acc.copy(),
            np.zeros(3, dtype=np.float64),
        ]

        try:
            ctrl_pts = UniformBSpline.parameterize_to_bspline(
                ts=ts,
                point_set=[path_world[i] for i in range(len(path_world))],
                start_end_derivative=start_end,
            )
            trajectory: Any = UniformBSpline(ctrl_pts, order=3, interval=ts)
            trajectory.set_physical_limits(self.cfg.max_vel, self.cfg.max_acc, self.cfg.feasibility_tolerance)
            feasible, ratio = trajectory.check_feasibility()
            if not feasible:
                trajectory.lengthen_time(max(ratio * 1.05, 1.05))
            if self._trajectory_collides(trajectory, current_pos, current_quat):
                raise RuntimeError("B-spline smoothing introduced a collision.")
            mode = "bspline"
        except Exception:
            trajectory = TimedPolyline(path_world, max_vel=self.cfg.max_vel * 0.9)
            mode = "polyline"

        duration = max(float(trajectory.get_time_sum()), self.cfg.min_plan_duration_s)
        return LocalPlan(
            trajectory=trajectory,
            start_time=float(sim_time),
            duration=duration,
            goal_world=goal_world.copy(),
            local_target_world=local_target_world.copy(),
            path_world=path_world.copy(),
            path_local=path_local.copy(),
            mode=mode,
            metadata={
                "reason": reason,
                "num_waypoints": int(len(path_world)),
                "local_target_local": local_target_local.copy(),
            },
        )

    def should_replan(
        self,
        sim_time: float,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        current_plan: LocalPlan | None,
    ) -> tuple[bool, str]:
        if current_plan is None:
            return True, "bootstrap"
        age = float(sim_time - current_plan.start_time)
        if age >= self.cfg.replan_period_s:
            return True, "periodic"
        if current_plan.remaining_time(sim_time) <= self.cfg.min_remaining_s:
            return True, "plan_ending"
        if self.goal_reached(current_pos, current_plan.goal_world):
            return False, "goal_close"
        if self._trajectory_collides(current_plan.trajectory, current_pos, current_quat, from_time=max(age, 0.0)):
            return True, "collision_predicted"
        return False, "keep"
