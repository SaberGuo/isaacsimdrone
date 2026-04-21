from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _as_vec3(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=np.float64).reshape(3)
    return arr


def _clamp_time(t: float, t_min: float, t_max: float) -> float:
    if t <= t_min:
        return float(t_min)
    if t >= t_max:
        return float(t_max)
    return float(t)


@dataclass
class TrajectorySample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


class PolynomialTrajectory:
    """Port of the EGO-planner polynomial trajectory utilities."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.cxs: list[list[float]] = []
        self.cys: list[list[float]] = []
        self.czs: list[list[float]] = []
        self.time_sum: float = 0.0
        self.num_seg: int = 0

    def reset(self) -> None:
        self.times.clear()
        self.cxs.clear()
        self.cys.clear()
        self.czs.clear()
        self.time_sum = 0.0
        self.num_seg = 0

    def add_segment(
        self,
        cx: Iterable[float],
        cy: Iterable[float],
        cz: Iterable[float],
        t: float,
    ) -> None:
        self.cxs.append([float(v) for v in cx])
        self.cys.append([float(v) for v in cy])
        self.czs.append([float(v) for v in cz])
        self.times.append(float(t))
        self.init()

    def init(self) -> None:
        self.num_seg = len(self.times)
        self.time_sum = float(sum(self.times))

    def get_time_sum(self) -> float:
        return float(self.time_sum)

    def _segment_at(self, t: float) -> tuple[int, float]:
        if self.num_seg == 0:
            raise RuntimeError("PolynomialTrajectory has no segments.")
        if self.num_seg == 1:
            return 0, _clamp_time(t, 0.0, self.times[0])
        local_t = _clamp_time(t, 0.0, max(self.time_sum - 1.0e-6, 0.0))
        idx = 0
        while idx < self.num_seg - 1 and self.times[idx] + 1.0e-4 < local_t:
            local_t -= self.times[idx]
            idx += 1
        local_t = _clamp_time(local_t, 0.0, self.times[idx])
        return idx, local_t

    def evaluate(self, t: float) -> np.ndarray:
        idx, local_t = self._segment_at(t)
        order = len(self.cxs[idx])
        tv = np.array(
            [local_t ** (order - 1 - i) for i in range(order)],
            dtype=np.float64,
        )
        pos = np.array(
            [
                tv.dot(np.asarray(self.cxs[idx], dtype=np.float64)),
                tv.dot(np.asarray(self.cys[idx], dtype=np.float64)),
                tv.dot(np.asarray(self.czs[idx], dtype=np.float64)),
            ],
            dtype=np.float64,
        )
        return pos

    def evaluate_vel(self, t: float) -> np.ndarray:
        idx, local_t = self._segment_at(t)
        order = len(self.cxs[idx])
        vx = np.array(
            [(i + 1) * self.cxs[idx][order - 2 - i] for i in range(order - 1)],
            dtype=np.float64,
        )
        vy = np.array(
            [(i + 1) * self.cys[idx][order - 2 - i] for i in range(order - 1)],
            dtype=np.float64,
        )
        vz = np.array(
            [(i + 1) * self.czs[idx][order - 2 - i] for i in range(order - 1)],
            dtype=np.float64,
        )
        tv = np.array([local_t ** i for i in range(order - 1)], dtype=np.float64)
        vel = np.array([tv.dot(vx), tv.dot(vy), tv.dot(vz)], dtype=np.float64)
        return vel

    def evaluate_acc(self, t: float) -> np.ndarray:
        idx, local_t = self._segment_at(t)
        order = len(self.cxs[idx])
        ax = np.array(
            [
                (i + 2) * (i + 1) * self.cxs[idx][order - 3 - i]
                for i in range(order - 2)
            ],
            dtype=np.float64,
        )
        ay = np.array(
            [
                (i + 2) * (i + 1) * self.cys[idx][order - 3 - i]
                for i in range(order - 2)
            ],
            dtype=np.float64,
        )
        az = np.array(
            [
                (i + 2) * (i + 1) * self.czs[idx][order - 3 - i]
                for i in range(order - 2)
            ],
            dtype=np.float64,
        )
        tv = np.array([local_t ** i for i in range(order - 2)], dtype=np.float64)
        acc = np.array([tv.dot(ax), tv.dot(ay), tv.dot(az)], dtype=np.float64)
        return acc

    def sample(self, t: float) -> TrajectorySample:
        return TrajectorySample(
            position=self.evaluate(t),
            velocity=self.evaluate_vel(t),
            acceleration=self.evaluate_acc(t),
        )

    @staticmethod
    def one_segment_traj_gen(
        start_pt: Iterable[float],
        start_vel: Iterable[float],
        start_acc: Iterable[float],
        end_pt: Iterable[float],
        end_vel: Iterable[float],
        end_acc: Iterable[float],
        t: float,
    ) -> "PolynomialTrajectory":
        t = max(float(t), 1.0e-3)
        c = np.zeros((6, 6), dtype=np.float64)
        c[0, 5] = 1.0
        c[1, 4] = 1.0
        c[2, 3] = 2.0
        c[3] = np.array([t**5, t**4, t**3, t**2, t, 1.0], dtype=np.float64)
        c[4] = np.array([5 * t**4, 4 * t**3, 3 * t**2, 2 * t, 1.0, 0.0], dtype=np.float64)
        c[5] = np.array([20 * t**3, 12 * t**2, 6 * t, 2.0, 0.0, 0.0], dtype=np.float64)

        bx = np.array(
            [
                _as_vec3(start_pt)[0],
                _as_vec3(start_vel)[0],
                _as_vec3(start_acc)[0],
                _as_vec3(end_pt)[0],
                _as_vec3(end_vel)[0],
                _as_vec3(end_acc)[0],
            ],
            dtype=np.float64,
        )
        by = np.array(
            [
                _as_vec3(start_pt)[1],
                _as_vec3(start_vel)[1],
                _as_vec3(start_acc)[1],
                _as_vec3(end_pt)[1],
                _as_vec3(end_vel)[1],
                _as_vec3(end_acc)[1],
            ],
            dtype=np.float64,
        )
        bz = np.array(
            [
                _as_vec3(start_pt)[2],
                _as_vec3(start_vel)[2],
                _as_vec3(start_acc)[2],
                _as_vec3(end_pt)[2],
                _as_vec3(end_vel)[2],
                _as_vec3(end_acc)[2],
            ],
            dtype=np.float64,
        )

        coef_x = np.linalg.solve(c, bx)
        coef_y = np.linalg.solve(c, by)
        coef_z = np.linalg.solve(c, bz)

        traj = PolynomialTrajectory()
        traj.add_segment(coef_x.tolist(), coef_y.tolist(), coef_z.tolist(), t)
        return traj

    @staticmethod
    def min_snap_traj(
        pos: np.ndarray,
        start_vel: Iterable[float],
        end_vel: Iterable[float],
        start_acc: Iterable[float],
        end_acc: Iterable[float],
        time_alloc: Iterable[float],
    ) -> "PolynomialTrajectory":
        time_vec = np.asarray(list(time_alloc), dtype=np.float64).reshape(-1)
        seg_num = int(time_vec.size)
        if seg_num <= 1:
            return PolynomialTrajectory.one_segment_traj_gen(
                pos[:, 0],
                start_vel,
                start_acc,
                pos[:, -1],
                end_vel,
                end_acc,
                float(time_vec[0]) if seg_num == 1 else 1.0,
            )

        poly_coeff = np.zeros((seg_num, 18), dtype=np.float64)
        px = np.zeros(6 * seg_num, dtype=np.float64)
        py = np.zeros(6 * seg_num, dtype=np.float64)
        pz = np.zeros(6 * seg_num, dtype=np.float64)

        dx = np.zeros(seg_num * 6, dtype=np.float64)
        dy = np.zeros(seg_num * 6, dtype=np.float64)
        dz = np.zeros(seg_num * 6, dtype=np.float64)

        start_vel = _as_vec3(start_vel)
        end_vel = _as_vec3(end_vel)
        start_acc = _as_vec3(start_acc)
        end_acc = _as_vec3(end_acc)

        for k in range(seg_num):
            dx[k * 6 + 0] = pos[0, k]
            dx[k * 6 + 1] = pos[0, k + 1]
            dy[k * 6 + 0] = pos[1, k]
            dy[k * 6 + 1] = pos[1, k + 1]
            dz[k * 6 + 0] = pos[2, k]
            dz[k * 6 + 1] = pos[2, k + 1]

            if k == 0:
                dx[k * 6 + 2] = start_vel[0]
                dy[k * 6 + 2] = start_vel[1]
                dz[k * 6 + 2] = start_vel[2]
                dx[k * 6 + 4] = start_acc[0]
                dy[k * 6 + 4] = start_acc[1]
                dz[k * 6 + 4] = start_acc[2]
            elif k == seg_num - 1:
                dx[k * 6 + 3] = end_vel[0]
                dy[k * 6 + 3] = end_vel[1]
                dz[k * 6 + 3] = end_vel[2]
                dx[k * 6 + 5] = end_acc[0]
                dy[k * 6 + 5] = end_acc[1]
                dz[k * 6 + 5] = end_acc[2]

        a = np.zeros((seg_num * 6, seg_num * 6), dtype=np.float64)
        for k in range(seg_num):
            ab = np.zeros((6, 6), dtype=np.float64)
            for i in range(3):
                ab[2 * i, i] = math.factorial(i)
                for j in range(i, 6):
                    ab[2 * i + 1, j] = (
                        math.factorial(j) / math.factorial(j - i) * (time_vec[k] ** (j - i))
                    )
            a[k * 6 : (k + 1) * 6, k * 6 : (k + 1) * 6] = ab

        num_f = 2 * seg_num + 4
        num_p = 2 * seg_num - 2
        num_d = 6 * seg_num
        ct = np.zeros((num_d, num_f + num_p), dtype=np.float64)
        ct[0, 0] = 1.0
        ct[2, 1] = 1.0
        ct[4, 2] = 1.0
        ct[1, 3] = 1.0
        ct[3, 2 * seg_num + 4] = 1.0
        ct[5, 2 * seg_num + 5] = 1.0

        ct[6 * (seg_num - 1) + 0, 2 * seg_num + 0] = 1.0
        ct[6 * (seg_num - 1) + 1, 2 * seg_num + 1] = 1.0
        ct[6 * (seg_num - 1) + 2, 4 * seg_num + 0] = 1.0
        ct[6 * (seg_num - 1) + 3, 2 * seg_num + 2] = 1.0
        ct[6 * (seg_num - 1) + 4, 4 * seg_num + 1] = 1.0
        ct[6 * (seg_num - 1) + 5, 2 * seg_num + 3] = 1.0

        for j in range(2, seg_num):
            ct[6 * (j - 1) + 0, 2 + 2 * (j - 1) + 0] = 1.0
            ct[6 * (j - 1) + 1, 2 + 2 * (j - 1) + 1] = 1.0
            ct[6 * (j - 1) + 2, 2 * seg_num + 4 + 2 * (j - 2) + 0] = 1.0
            ct[6 * (j - 1) + 3, 2 * seg_num + 4 + 2 * (j - 1) + 0] = 1.0
            ct[6 * (j - 1) + 4, 2 * seg_num + 4 + 2 * (j - 2) + 1] = 1.0
            ct[6 * (j - 1) + 5, 2 * seg_num + 4 + 2 * (j - 1) + 1] = 1.0

        c = ct.T
        dx1 = c @ dx
        dy1 = c @ dy
        dz1 = c @ dz

        q = np.zeros((seg_num * 6, seg_num * 6), dtype=np.float64)
        for k in range(seg_num):
            for i in range(3, 6):
                for j in range(3, 6):
                    q[k * 6 + i, k * 6 + j] = (
                        i
                        * (i - 1)
                        * (i - 2)
                        * j
                        * (j - 1)
                        * (j - 2)
                        / (i + j - 5)
                        * (time_vec[k] ** (i + j - 5))
                    )

        a_inv = np.linalg.inv(a)
        r = c @ a_inv.T @ q @ a_inv @ ct

        dxf = dx1[:num_f]
        dyf = dy1[:num_f]
        dzf = dz1[:num_f]

        rfp = r[:num_f, num_f:]
        rpp = r[num_f:, num_f:]

        dxp = -(np.linalg.solve(rpp, rfp.T @ dxf))
        dyp = -(np.linalg.solve(rpp, rfp.T @ dyf))
        dzp = -(np.linalg.solve(rpp, rfp.T @ dzf))

        dx1[num_f:] = dxp
        dy1[num_f:] = dyp
        dz1[num_f:] = dzp

        px = (a_inv @ ct) @ dx1
        py = (a_inv @ ct) @ dy1
        pz = (a_inv @ ct) @ dz1

        for i in range(seg_num):
            poly_coeff[i, 0:6] = px[i * 6 : i * 6 + 6]
            poly_coeff[i, 6:12] = py[i * 6 : i * 6 + 6]
            poly_coeff[i, 12:18] = pz[i * 6 : i * 6 + 6]

        poly_traj = PolynomialTrajectory()
        for i in range(poly_coeff.shape[0]):
            cx = list(reversed(poly_coeff[i, 0:6].tolist()))
            cy = list(reversed(poly_coeff[i, 6:12].tolist()))
            cz = list(reversed(poly_coeff[i, 12:18].tolist()))
            poly_traj.add_segment(cx, cy, cz, float(time_vec[i]))
        return poly_traj


class TimedPolyline:
    """Fallback trajectory that time-parameterizes a waypoint polyline."""

    def __init__(self, points: np.ndarray, max_vel: float) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ValueError("TimedPolyline expects shape (N, 3) with N >= 2.")
        self.points = pts
        self.max_vel = max(float(max_vel), 1.0e-3)
        seg_vec = self.points[1:] - self.points[:-1]
        seg_len = np.linalg.norm(seg_vec, axis=1)
        seg_time = np.maximum(seg_len / self.max_vel, 1.0e-2)
        self.seg_len = seg_len
        self.seg_time = seg_time
        self.cum_time = np.concatenate([[0.0], np.cumsum(seg_time)])
        self.duration = float(self.cum_time[-1])

    def get_time_sum(self) -> float:
        return float(self.duration)

    def sample(self, t: float) -> TrajectorySample:
        t = _clamp_time(t, 0.0, self.duration)
        idx = int(np.searchsorted(self.cum_time, t, side="right") - 1)
        idx = min(max(idx, 0), len(self.seg_time) - 1)
        t0 = self.cum_time[idx]
        seg_t = max(self.seg_time[idx], 1.0e-3)
        alpha = np.clip((t - t0) / seg_t, 0.0, 1.0)
        p0 = self.points[idx]
        p1 = self.points[idx + 1]
        vel = (p1 - p0) / seg_t
        pos = (1.0 - alpha) * p0 + alpha * p1
        acc = np.zeros(3, dtype=np.float64)
        return TrajectorySample(position=pos, velocity=vel, acceleration=acc)


class UniformBSpline:
    """Port of the EGO-planner uniform B-spline implementation."""

    def __init__(self, points: np.ndarray, order: int, interval: float) -> None:
        self.control_points = np.zeros((3, 0), dtype=np.float64)
        self.order = int(order)
        self.n = -1
        self.m = -1
        self.knots = np.zeros((0,), dtype=np.float64)
        self.interval = float(interval)
        self.limit_vel = 0.0
        self.limit_acc = 0.0
        self.limit_ratio = 1.1
        self.feasibility_tolerance = 0.0
        self.set_uniform_bspline(points, order, interval)

    def set_uniform_bspline(self, points: np.ndarray, order: int, interval: float) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] != 3:
            raise ValueError("UniformBSpline expects control points with shape (3, N).")
        self.control_points = pts.copy()
        self.order = int(order)
        self.interval = float(interval)
        self.n = pts.shape[1] - 1
        self.m = self.n + self.order + 1
        self.knots = np.zeros((self.m + 1,), dtype=np.float64)
        for i in range(self.m + 1):
            if i <= self.order:
                self.knots[i] = float(-self.order + i) * self.interval
            else:
                self.knots[i] = self.knots[i - 1] + self.interval

    def set_knot(self, knot: np.ndarray) -> None:
        self.knots = np.asarray(knot, dtype=np.float64).copy()

    def get_knot(self) -> np.ndarray:
        return self.knots.copy()

    def get_control_points(self) -> np.ndarray:
        return self.control_points.copy()

    def get_time_span(self) -> tuple[float, float]:
        if self.order >= self.knots.shape[0] or self.m - self.order >= self.knots.shape[0]:
            raise RuntimeError("Invalid knot vector for B-spline.")
        return float(self.knots[self.order]), float(self.knots[self.m - self.order])

    def evaluate_deboor(self, u: float) -> np.ndarray:
        u_min, u_max = self.get_time_span()
        ub = _clamp_time(u, u_min, u_max)
        k = self.order
        while k + 1 < self.knots.size and self.knots[k + 1] < ub:
            k += 1
        d = [
            self.control_points[:, k - self.order + i].copy()
            for i in range(self.order + 1)
        ]
        for r in range(1, self.order + 1):
            for i in range(self.order, r - 1, -1):
                denom = self.knots[i + 1 + k - r] - self.knots[i + k - self.order]
                alpha = 0.0 if abs(denom) < 1.0e-9 else (ub - self.knots[i + k - self.order]) / denom
                d[i] = (1.0 - alpha) * d[i - 1] + alpha * d[i]
        return d[self.order]

    def evaluate_deboor_t(self, t: float) -> np.ndarray:
        return self.evaluate_deboor(float(t) + self.knots[self.order])

    def get_derivative_control_points(self) -> np.ndarray:
        ctp = np.zeros((self.control_points.shape[0], self.control_points.shape[1] - 1), dtype=np.float64)
        for i in range(ctp.shape[1]):
            denom = self.knots[i + self.order + 1] - self.knots[i + 1]
            ctp[:, i] = self.order * (self.control_points[:, i + 1] - self.control_points[:, i]) / denom
        return ctp

    def get_derivative(self) -> "UniformBSpline":
        ctp = self.get_derivative_control_points()
        derivative = UniformBSpline(ctp, self.order - 1, self.interval)
        derivative.set_knot(self.knots[1:-1])
        return derivative

    def set_physical_limits(self, vel: float, acc: float, tolerance: float) -> None:
        self.limit_vel = float(vel)
        self.limit_acc = float(acc)
        self.limit_ratio = 1.1
        self.feasibility_tolerance = float(tolerance)

    def check_feasibility(self) -> tuple[bool, float]:
        p = self.control_points
        dim = p.shape[0]
        feasible = True

        max_vel = -1.0
        vel_limit = self.limit_vel * (1.0 + self.feasibility_tolerance) + 1.0e-4
        for i in range(p.shape[1] - 1):
            vel = self.order * (p[:, i + 1] - p[:, i]) / (self.knots[i + self.order + 1] - self.knots[i + 1])
            if np.any(np.abs(vel) > vel_limit):
                feasible = False
                max_vel = max(max_vel, float(np.max(np.abs(vel[:dim]))))

        max_acc = -1.0
        acc_limit = self.limit_acc * (1.0 + self.feasibility_tolerance) + 1.0e-4
        for i in range(p.shape[1] - 2):
            acc = (
                self.order
                * (self.order - 1)
                * (
                    (p[:, i + 2] - p[:, i + 1]) / (self.knots[i + self.order + 2] - self.knots[i + 2])
                    - (p[:, i + 1] - p[:, i]) / (self.knots[i + self.order + 1] - self.knots[i + 1])
                )
                / (self.knots[i + self.order + 1] - self.knots[i + 2])
            )
            if np.any(np.abs(acc) > acc_limit):
                feasible = False
                max_acc = max(max_acc, float(np.max(np.abs(acc[:dim]))))

        if max_vel < 0.0 and max_acc < 0.0:
            return True, 1.0
        ratio = max(
            max(max_vel / max(self.limit_vel, 1.0e-6), 1.0),
            max(math.sqrt(abs(max_acc) / max(self.limit_acc, 1.0e-6)), 1.0),
        )
        return feasible, float(ratio)

    def lengthen_time(self, ratio: float) -> None:
        ratio = max(float(ratio), 1.0)
        num1 = 5
        num2 = self.knots.shape[0] - 1 - 5
        if num2 <= num1:
            return
        delta_t = (ratio - 1.0) * (self.knots[num2] - self.knots[num1])
        t_inc = delta_t / float(num2 - num1)
        for i in range(num1 + 1, num2 + 1):
            self.knots[i] += float(i - num1) * t_inc
        for i in range(num2 + 1, self.knots.shape[0]):
            self.knots[i] += delta_t

    def get_time_sum(self) -> float:
        u_min, u_max = self.get_time_span()
        return float(u_max - u_min)

    def sample(self, t: float) -> TrajectorySample:
        pos = self.evaluate_deboor_t(t)
        vel = self.get_derivative().evaluate_deboor_t(t)
        acc = self.get_derivative().get_derivative().evaluate_deboor_t(t)
        return TrajectorySample(position=pos, velocity=vel, acceleration=acc)

    @staticmethod
    def parameterize_to_bspline(
        ts: float,
        point_set: list[np.ndarray],
        start_end_derivative: list[np.ndarray],
    ) -> np.ndarray:
        if ts <= 0.0:
            raise ValueError("B-spline time step must be positive.")
        if len(point_set) <= 3:
            raise ValueError("B-spline parameterization requires at least 4 points.")
        if len(start_end_derivative) != 4:
            raise ValueError("start_end_derivative must contain [v0, v1, a0, a1].")

        k = len(point_set)
        prow = np.array([1.0, 4.0, 1.0], dtype=np.float64)
        vrow = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        arow = np.array([1.0, -2.0, 1.0], dtype=np.float64)

        a = np.zeros((k + 4, k + 2), dtype=np.float64)
        for i in range(k):
            a[i, i : i + 3] = (1.0 / 6.0) * prow
        a[k, 0:3] = (1.0 / (2.0 * ts)) * vrow
        a[k + 1, k - 1 : k + 2] = (1.0 / (2.0 * ts)) * vrow
        a[k + 2, 0:3] = (1.0 / (ts * ts)) * arow
        a[k + 3, k - 1 : k + 2] = (1.0 / (ts * ts)) * arow

        b = np.zeros((k + 4, 3), dtype=np.float64)
        for i, p in enumerate(point_set):
            b[i] = _as_vec3(p)
        for i, d in enumerate(start_end_derivative):
            b[k + i] = _as_vec3(d)

        ctrl_pts, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
        return ctrl_pts.T


def densify_polyline(points: np.ndarray, spacing: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= 1:
        return pts.copy()

    spacing = max(float(spacing), 1.0e-3)
    dense: list[np.ndarray] = [pts[0].copy()]
    for idx in range(len(pts) - 1):
        p0 = pts[idx]
        p1 = pts[idx + 1]
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1.0e-9:
            continue
        steps = max(int(math.ceil(length / spacing)), 1)
        for step in range(1, steps + 1):
            alpha = step / float(steps)
            dense.append((1.0 - alpha) * p0 + alpha * p1)
    return np.asarray(dense, dtype=np.float64)


def ensure_min_points(points: np.ndarray, min_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) >= int(min_points):
        return pts.copy()
    if len(pts) < 2:
        raise ValueError("Need at least 2 points to interpolate.")

    target = max(int(min_points), len(pts))
    new_points: list[np.ndarray] = []
    total_segments = len(pts) - 1
    for i in range(total_segments):
        p0 = pts[i]
        p1 = pts[i + 1]
        local_steps = max(int(math.ceil((target - 1) / total_segments)), 1)
        for step in range(local_steps):
            alpha = step / float(local_steps)
            new_points.append((1.0 - alpha) * p0 + alpha * p1)
    new_points.append(pts[-1].copy())
    out = np.asarray(new_points, dtype=np.float64)
    if len(out) >= target:
        return out[:target]
    return ensure_min_points(out, target)

