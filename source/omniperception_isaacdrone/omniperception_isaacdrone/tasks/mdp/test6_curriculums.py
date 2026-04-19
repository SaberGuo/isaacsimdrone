from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _init_curriculum_state(
    env: "ManagerBasedRLEnv",
    levels: Sequence[int],
    num_envs: int,
    k_roll: int,
    device: torch.device,
) -> None:
    u = env.unwrapped
    window_size = num_envs * k_roll

    u.curr_levels       = list(levels)
    u.curr_num_envs     = int(num_envs)
    u.curr_k_roll       = int(k_roll)
    u.curr_window_size  = int(window_size)
    u.curr_device       = device

    # 循环缓冲区：shape=(window_size,)，-1 表示未填写
    u.curr_history = torch.full(
        (window_size,), -1.0, dtype=torch.float32, device=device
    )

    # 每个 env 下一次写入的槽位偏移（在各自的 k_roll 个槽内循环）
    u.curr_write_round = torch.zeros(num_envs, dtype=torch.long, device=device)

    u.curr_level_idx = 0

    # ------------------------------------------------------------------ #
    # 统一字段名：curr_obstacle_count（test6_events.py 也读取此字段）       #
    # ------------------------------------------------------------------ #
    u.curr_obstacle_count = int(levels[0])

    # 触发首次障碍物布置
    u.obstacle_level_changed = True

    print(
        f"[CURRICULUM] 初始化: levels={list(levels)}, "
        f"num_envs={num_envs}, k_roll={k_roll}, "
        f"window_size={window_size}, "
        f"初始障碍物数量={u.curr_obstacle_count}",
        flush=True,
    )


def _compute_success_rate(env: "ManagerBasedRLEnv") -> float:
    u = env.unwrapped
    valid_mask = u.curr_history >= 0.0
    if not valid_mask.any():
        return 0.0
    return float(u.curr_history[valid_mask].mean().item())


def _window_is_full(env: "ManagerBasedRLEnv") -> bool:
    """
    窗口满的判断：有效槽位数（已写入过至少一次的槽）达到 window_size。
    不依赖每个 env 的独立计数，避免长 episode 的 env 永远不 reset
    导致 .all() 无法为 True 的问题。
    """
    u = env.unwrapped
    filled = int((u.curr_history >= 0.0).sum().item())
    return filled >= u.curr_window_size


def _resolve_success_threshold(
    env: "ManagerBasedRLEnv",
    success_threshold: float,
    success_thresholds: Sequence[float] | None,
) -> float:
    u = env.unwrapped
    if not success_thresholds:
        return float(success_threshold)

    thresholds = list(success_thresholds)
    if len(thresholds) == 0:
        return float(success_threshold)

    level_idx = int(getattr(u, "curr_level_idx", 0))
    level_idx = max(0, min(level_idx, len(thresholds) - 1))
    return float(thresholds[level_idx])


def update_obstacle_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | slice | torch.Tensor,
    levels: Sequence[int] = (0, 10, 20, 30, 50, 100),
    success_term_name: str = "reached_goal",
    success_threshold: float = 0.85,
    success_thresholds: Sequence[float] | None = None,
    window_size: int = 200,   # 保留签名兼容性，实际由 num_envs * k_roll 决定
    k_roll: int = 4,
) -> dict[str, float]:
    u = env.unwrapped

    # 延迟初始化
    if not hasattr(u, "curr_history"):
        num_envs = int(u.num_envs)
        device   = torch.device(getattr(u, "device", "cpu"))
        _init_curriculum_state(env, levels, num_envs, k_roll, device)

    device = u.curr_device

    # ------------------------------------------------------------------
    # 构造默认日志（用于 env_ids 为空时的早返回）
    # ------------------------------------------------------------------
    def _default_log() -> dict[str, float]:
        filled = int((u.curr_history >= 0.0).sum().item())
        current_threshold = _resolve_success_threshold(env, success_threshold, success_thresholds)
        return {
            "Curriculum/active_count":        float(u.curr_obstacle_count),
            "Curriculum/level_idx":           float(u.curr_level_idx),
            "Curriculum/success_rate":        _compute_success_rate(env),
            "Curriculum/success_threshold":   float(current_threshold),
            "Curriculum/window_full":         float(_window_is_full(env)),
            "Curriculum/window_filled_slots": float(filled),
            "Curriculum/window_size":         float(u.curr_window_size),
        }

    # ------------------------------------------------------------------
    # 解析 env_ids
    # ------------------------------------------------------------------
    if env_ids is None or (
        isinstance(env_ids, torch.Tensor) and env_ids.numel() == 0
    ):
        return _default_log()

    if isinstance(env_ids, slice):
        ids = torch.arange(
            *env_ids.indices(u.curr_num_envs), dtype=torch.long, device=device
        )
    elif isinstance(env_ids, torch.Tensor):
        ids = env_ids.to(dtype=torch.long, device=device).reshape(-1)
    else:
        ids = torch.tensor(list(env_ids), dtype=torch.long, device=device)

    if ids.numel() == 0:
        return _default_log()

    # ------------------------------------------------------------------
    # 获取本次 reset 的 env 的成功标志
    # ------------------------------------------------------------------
    success_term = env.termination_manager.get_term(success_term_name)
    if success_term is None:
        return _default_log()

    successes = success_term[ids].float().to(device)

    # ------------------------------------------------------------------
    # 写入循环缓冲区
    # env_i 的槽位为 [i, i+num_envs, i+2*num_envs, ...]（共 k_roll 个）
    # curr_write_round[i] 记录下一次写第几个槽（0 ~ k_roll-1 循环）
    # ------------------------------------------------------------------
    current_rounds = u.curr_write_round[ids]               # (len(ids),)
    slots = ids + current_rounds * u.curr_num_envs          # 全局槽位索引
    u.curr_history.scatter_(0, slots, successes)
    u.curr_write_round[ids] = (current_rounds + 1) % u.curr_k_roll

    # ------------------------------------------------------------------
    # 判断是否晋级
    # ------------------------------------------------------------------
    success_rate = _compute_success_rate(env)
    is_full      = _window_is_full(env)
    current_threshold = _resolve_success_threshold(env, success_threshold, success_thresholds)

    if (
        is_full
        and success_rate >= current_threshold
        and u.curr_level_idx < len(u.curr_levels) - 1
    ):
        u.curr_level_idx      += 1
        # ----------------------------------------------------------------
        # 关键修复：同时更新 curr_obstacle_count
        # test6_events.py 读取的正是这个字段
        # ----------------------------------------------------------------
        u.curr_obstacle_count = int(u.curr_levels[u.curr_level_idx])

        # 晋级后清空历史，write_round 保持不变（继续从各自下一个槽写入）
        u.curr_history.fill_(-1.0)

        # 通知 Event 重排障碍物
        u.obstacle_level_changed = True

        print(
            f"[CURRICULUM] 成功率 {success_rate:.3f} >= {current_threshold:.3f} | "
            f"晋级 → 难度等级 {u.curr_level_idx} | "
            f"障碍物数量: {u.curr_obstacle_count}",
            flush=True,
        )

        success_rate = 0.0
        is_full      = False
        current_threshold = _resolve_success_threshold(env, success_threshold, success_thresholds)

    filled = int((u.curr_history >= 0.0).sum().item())
    return {
        "Curriculum/active_count":        float(u.curr_obstacle_count),
        "Curriculum/level_idx":           float(u.curr_level_idx),
        "Curriculum/success_rate":        float(success_rate),
        "Curriculum/success_threshold":   float(current_threshold),
        "Curriculum/window_full":         float(is_full),
        "Curriculum/window_filled_slots": float(filled),
        "Curriculum/window_size":         float(u.curr_window_size),
    }
