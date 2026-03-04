# SPDX-License-Identifier: BSD-3-Clause
"""
LiDAR Visualizer Module - Minimal Version
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:
    torch = None


@dataclass
class LidarSnapshot:
    """激光雷达数据快照"""
    env_idx: int
    step: int
    timestamp: str
    sensor_pos: np.ndarray
    sensor_quat: np.ndarray
    distances: np.ndarray
    num_rays: int
    num_valid: int
    valid_ratio: float
    dist_min: float
    dist_max: float
    dist_mean: float
    dist_std: float
    pointcloud: Optional[np.ndarray] = None
    max_distance: float = 20.0
    min_range: float = 0.2


def _to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return np.array([])
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def get_lidar_data(
    lidar: Any,
    env_idx: int = 0,
    step: int = 0,
    include_pointcloud: bool = False,
) -> Optional[LidarSnapshot]:
    """提取激光雷达数据快照"""
    
    max_distance = 20.0
    min_range = 0.2
    
    if hasattr(lidar, "cfg"):
        cfg = lidar.cfg
        if hasattr(cfg, "max_distance"):
            max_distance = float(cfg.max_distance)
        if hasattr(cfg, "min_range"):
            min_range = float(cfg.min_range)
    
    if not hasattr(lidar, "data"):
        return None
    
    data = lidar.data
    distances = None
    
    for attr in ["ray_distances", "distances", "ranges", "distance", "range"]:
        if hasattr(data, attr):
            val = getattr(data, attr)
            if val is not None:
                distances = _to_numpy(val)
                break
    
    if distances is None or distances.size == 0:
        return None
    
    if distances.ndim == 2:
        if env_idx < distances.shape[0]:
            distances = distances[env_idx]
        else:
            distances = distances[0]
    
    # 位置 - 修复：使用 np.array 而不是 np.zeros
    pos = np.array([0.0, 0.0, 0.0])
    
    if hasattr(data, "pos_w"):
        pos_arr = _to_numpy(data.pos_w)
        if pos_arr.ndim == 2 and env_idx < pos_arr.shape[0]:
            pos = pos_arr[env_idx]
        elif pos_arr.ndim == 1 and pos_arr.shape[0] >= 3:
            pos = pos_arr[:3]
    
    # 四元数 - 修复：添加 quat 变量定义
    quat = np.array([1.0, 0.0, 0.0, 0.0])  # 默认单位四元数 (w, x, y, z)
    
    if hasattr(data, "quat_w"):
        quat_arr = _to_numpy(data.quat_w)
        if quat_arr.ndim == 2 and env_idx < quat_arr.shape[0]:
            quat = quat_arr[env_idx]
        elif quat_arr.ndim == 1 and quat_arr.shape[0] >= 4:
            quat = quat_arr[:4]
    
    # 统计
    valid_mask = (distances >= min_range) & (distances < max_distance - 0.01)
    valid_distances = distances[valid_mask]
    
    num_rays = int(distances.shape[0])
    num_valid = int(valid_mask.sum())
    valid_ratio = float(num_valid) / max(num_rays, 1)
    
    if num_valid > 0:
        dist_min = float(valid_distances.min())
        dist_max = float(valid_distances.max())
        dist_mean = float(valid_distances.mean())
        dist_std = float(valid_distances.std())
    else:
        dist_min = dist_max = dist_mean = dist_std = 0.0
    
    return LidarSnapshot(
        env_idx=env_idx,
        step=step,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sensor_pos=pos,
        sensor_quat=quat,
        distances=distances,
        num_rays=num_rays,
        num_valid=num_valid,
        valid_ratio=valid_ratio,
        dist_min=dist_min,
        dist_max=dist_max,
        dist_mean=dist_mean,
        dist_std=dist_std,
        max_distance=max_distance,
        min_range=min_range,
    )


def format_lidar_stats(
    snapshot: LidarSnapshot,
    show_histogram: bool = True,
    histogram_bins: int = 10,
    histogram_width: int = 40,
) -> str:
    """格式化激光雷达统计信息"""
    lines = []
    sep = "=" * 80
    
    pos_str = f"({snapshot.sensor_pos[0]:.2f}, {snapshot.sensor_pos[1]:.2f}, {snapshot.sensor_pos[2]:.2f})"
    lines.append(sep)
    lines.append(f"[LiDAR] env={snapshot.env_idx} | step={snapshot.step} | pos={pos_str}")
    lines.append(f"  Rays: {snapshot.num_rays} | Valid: {snapshot.num_valid} ({snapshot.valid_ratio*100:.1f}%)")
    lines.append(f"  Dist: min={snapshot.dist_min:.2f}m, max={snapshot.dist_max:.2f}m, mean={snapshot.dist_mean:.2f}m")
    
    if show_histogram and snapshot.num_valid > 0:
        valid_mask = (snapshot.distances >= snapshot.min_range) & \
                     (snapshot.distances < snapshot.max_distance - 0.01)
        valid_distances = snapshot.distances[valid_mask]
        
        bin_edges = np.linspace(snapshot.min_range, snapshot.max_distance, histogram_bins + 1)
        hist, _ = np.histogram(valid_distances, bins=bin_edges)
        max_count = hist.max() if hist.max() > 0 else 1
        
        lines.append(f"  Histogram:")
        for i in range(histogram_bins):
            count = hist[i]
            bar_len = int(count / max_count * histogram_width)
            bar = "█" * bar_len
            lines.append(f"    [{bin_edges[i]:5.1f}-{bin_edges[i+1]:5.1f}): {bar} {count}")
    
    lines.append(sep)
    return "\n".join(lines)


def print_lidar_stats(
    lidar: Any,
    env_idx: int = 0,
    step: int = 0,
    show_histogram: bool = True,
    histogram_bins: int = 10,
) -> Optional[LidarSnapshot]:
    """打印激光雷达统计信息"""
    snapshot = get_lidar_data(lidar, env_idx=env_idx, step=step)
    
    if snapshot is None:
        print(f"[LiDAR] env={env_idx} | step={step} | ERROR: No data")
        return None
    
    print(format_lidar_stats(snapshot, show_histogram, histogram_bins))
    return snapshot


def save_lidar_snapshot(
    lidar: Any,
    env_idx: int = 0,
    step: int = 0,
    output_dir: str = "./lidar_snapshots",
    save_pointcloud: bool = True,
    save_distances: bool = True,
    save_stats: bool = True,
) -> dict:
    """保存激光雷达快照"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    snapshot = get_lidar_data(lidar, env_idx=env_idx, step=step)
    saved = {}
    
    if snapshot is None:
        return saved
    
    prefix = f"lidar_env{env_idx:03d}_step{step:06d}"
    
    if save_distances:
        p = output_dir / f"{prefix}_distances.npy"
        np.save(p, snapshot.distances)
        saved["distances"] = p
    
    if save_stats:
        p = output_dir / f"{prefix}_stats.txt"
        p.write_text(format_lidar_stats(snapshot))
        saved["stats"] = p
    
    return saved


def print_lidar_summary(lidars: list, step: int = 0) -> None:
    """打印所有环境的激光雷达摘要"""
    print(f"\n{'='*60}")
    print(f"[LiDAR Summary] step={step} | envs={len(lidars)}")
    for i, lidar in enumerate(lidars):
        snapshot = get_lidar_data(lidar, env_idx=0, step=step)
        if snapshot:
            print(f"  env={i}: valid={snapshot.num_valid}/{snapshot.num_rays}")
    print(f"{'='*60}\n")


def get_lidar_distances_tensor(lidar: Any, env_idx: int = 0):
    """获取距离 tensor"""
    if torch is None or not hasattr(lidar, "data"):
        return None
    
    data = lidar.data
    for attr in ["ray_distances", "distances", "ranges"]:
        if hasattr(data, attr):
            val = getattr(data, attr)
            if val is not None and torch.is_tensor(val):
                if val.ndim == 2 and env_idx < val.shape[0]:
                    return val[env_idx]
                return val
    return None
