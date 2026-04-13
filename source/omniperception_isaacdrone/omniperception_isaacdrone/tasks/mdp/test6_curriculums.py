from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _init_curriculum_state(
    env: "ManagerBasedRLEnv",
    levels: Sequence[int],
    window_size: int,
    num_envs: int,
    k_roll: int,
    device: torch.device,
) -> None:
    """初始化挂载在 env.unwrapped 上的课程状态张量。"""
    u = env.unwrapped
    u.curr_levels = list(levels)
    u.curr_num_envs = int(num_envs)
    u.curr_k_roll = int(k_roll)
    u.curr_window_size = int(window_size)
    u.curr_device = device
    u.curr_history = torch.full((window_size,), -1.0, dtype=torch.float32, device=device)
    u.curr_write_round = torch.zeros(num_envs, dtype=torch.long, device=device)
    u.curr_write_count = torch.zeros(num_envs, dtype=torch.long, device=device)
    u.curr_level_idx = 0
    u.curr_obstacle_count = int(levels[0])
    u.obstacle_level_changed = True


def _compute_success_rate(env: "ManagerBasedRLEnv") -> float:
    """计算有效槽位的平均成功率。"""
    u = env.unwrapped
    valid_mask = u.curr_history >= 0.0
    if not valid_mask.any():
        return 0.0
    return float(u.curr_history[valid_mask].mean().item())


def _window_is_full(env: "ManagerBasedRLEnv") -> bool:
    """判断所有 env 是否均已完成 k_roll 轮写入。"""
    u = env.unwrapped
    return bool((u.curr_write_count >= u.curr_k_roll).all().item())


def update_obstacle_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | slice | torch.Tensor,
    levels: Sequence[int] = (0, 10, 20, 40, 60, 100),
    success_term_name: str = "reached_goal",
    success_threshold: float = 0.8,
    window_size: int = 200,
    k_roll: int = 2,
) -> dict[str, float]:
    """基于固定槽位循环缓冲区更新障碍物课程等级。"""
    u = env.unwrapped
    if not hasattr(u, "curr_history"):
        num_envs = int(env.unwrapped.num_envs)
        device = torch.device(getattr(env.unwrapped, "device", "cpu"))
        actual_window = num_envs * k_roll
        _init_curriculum_state(env, levels, actual_window, num_envs, k_roll, device)
    device = u.curr_device

    def _default_log():
        return {
            "active_count": float(u.curr_obstacle_count),
            "level_idx": float(u.curr_level_idx),
            "success_rate": _compute_success_rate(env),
            "success_threshold": float(success_threshold),
            "window_full": float(_window_is_full(env)),
        }

    if env_ids is None or (isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0):
        return _default_log()

    if isinstance(env_ids, slice):
        ids = torch.arange(*env_ids.indices(u.curr_num_envs), dtype=torch.long, device=device)
    elif isinstance(env_ids, torch.Tensor):
        ids = env_ids.to(dtype=torch.long, device=device).reshape(-1)
    else:
        ids = torch.tensor(list(env_ids), dtype=torch.long, device=device)

    if ids.numel() == 0:
        return _default_log()

    success_term = env.termination_manager.get_term(success_term_name)
    if success_term is None:
        return _default_log()

    successes = success_term[ids].float().to(device)
    current_rounds = u.curr_write_round[ids]
    slots = ids + current_rounds * u.curr_num_envs
    u.curr_history.scatter_(0, slots, successes)
    u.curr_write_round[ids] = (current_rounds + 1) % u.curr_k_roll
    u.curr_write_count[ids] = u.curr_write_count[ids] + 1

    success_rate = _compute_success_rate(env)

    if (
        _window_is_full(env)
        and success_rate >= success_threshold
        and u.curr_level_idx < len(u.curr_levels) - 1
    ):
        u.curr_level_idx += 1
        u.curr_obstacle_count = int(u.curr_levels[u.curr_level_idx])
        u.curr_history.fill_(-1.0)
        u.curr_write_round.zero_()
        u.curr_write_count.zero_()
        u.obstacle_level_changed = True
        print(
            f"[CURRICULUM] 成功率 {success_rate:.3f} >= {success_threshold} | "
            f"晋级 → 难度等级 {u.curr_level_idx} | "
            f"障碍物数量: {u.curr_obstacle_count}",
            flush=True,
        )
        success_rate = 0.0

    return {
        "active_count": float(u.curr_obstacle_count),
        "level_idx": float(u.curr_level_idx),
        "success_rate": float(success_rate),
        "success_threshold": float(success_threshold),
        "window_full": float(_window_is_full(env)),
    }
