from __future__ import annotations

import math


def safe_float(value, default: float = 0.0) -> float:
    try:
        x = float(value)
    except Exception:
        return float(default)
    return x if math.isfinite(x) else float(default)


def safe_positive_resolution(resolution: float, default: float = 1.0, min_value: float = 1e-6) -> float:
    r = safe_float(resolution, default)
    if (not math.isfinite(r)) or abs(r) < float(min_value):
        return float(default)
    return abs(r)


def clip_int(value: int, low: int, high: int) -> int:
    return low if value < low else high if value > high else value


def safe_world_to_grid(
    x: float,
    y: float,
    *,
    x_min: float,
    y_min: float,
    resolution: float,
    width: int,
    height: int,
) -> tuple[int, int]:
    width_i = max(int(width), 1)
    height_i = max(int(height), 1)
    res = safe_positive_resolution(resolution)

    x0 = safe_float(x_min, 0.0)
    y0 = safe_float(y_min, 0.0)
    xf = safe_float(x, x0)
    yf = safe_float(y, y0)

    gx_f = (xf - x0) / res
    gy_f = (yf - y0) / res
    if not math.isfinite(gx_f):
        gx_f = 0.0
    if not math.isfinite(gy_f):
        gy_f = 0.0

    gx = int(round(gx_f))
    gy = int(round(gy_f))

    gx = clip_int(gx, 0, width_i - 1)
    gy = clip_int(gy, 0, height_i - 1)
    return gx, gy
