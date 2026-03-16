from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_env_ids(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor | slice | None,
) -> torch.Tensor:
    """Convert manager-provided env_ids to a 1-D long tensor on env.device."""
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long).reshape(-1)

    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long).reshape(-1)


def _is_step_driven_reset(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> bool:
    """Return True only for resets triggered by terminations inside env.step()."""
    reset_buf = getattr(env, "reset_buf", None)
    if not isinstance(reset_buf, torch.Tensor):
        return False
    if reset_buf.numel() != env.num_envs:
        return False
    if env_ids.numel() == 0:
        return False

    try:
        return bool(torch.all(reset_buf[env_ids]).item())
    except Exception:
        return False


def _ensure_history(env: "ManagerBasedRLEnv", window_size: int) -> deque:
    history = getattr(env, "_curriculum_success_window", None)

    if isinstance(history, deque) and history.maxlen == window_size:
        return history

    old_values = list(history) if isinstance(history, deque) else []
    history = deque(old_values[-window_size:], maxlen=window_size)
    setattr(env, "_curriculum_success_window", history)
    return history


def curriculum_obstacle_count_by_success(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor | slice | None,
    levels: Sequence[int] = (0, 10, 20, 40, 60, 100),
    initial_level: int = 0,
    success_term_name: str = "reached_goal",
    success_threshold: float = 0.8,
    window_size: int = 200,
    min_samples: int = 100,
    clear_history_on_promotion: bool = True,
) -> dict[str, float]:
    """Promote a global obstacle-count curriculum using recent success ratio.

    The term is evaluated by Isaac Lab at the beginning of ``env._reset_idx()``, i.e.
    immediately after the current step has been marked done and just before the reset
    is applied. This lets us:
      1. inspect the current batch of terminal outcomes; and
      2. change the shared obstacle count before the next episode starts.
    """
    levels = tuple(max(int(v), 0) for v in levels)
    if len(levels) == 0:
        levels = (0,)

    window_size = max(int(window_size), 1)
    min_samples = max(int(min_samples), 1)
    initial_level = max(0, min(int(initial_level), len(levels) - 1))
    success_threshold = float(success_threshold)

    needs_init = (
        tuple(getattr(env, "curriculum_obstacle_levels", ())) != levels
        or not isinstance(getattr(env, "_curriculum_success_window", None), deque)
        or getattr(getattr(env, "_curriculum_success_window", None), "maxlen", None) != window_size
    )
    if needs_init and hasattr(env, "configure_obstacle_curriculum"):
        env.configure_obstacle_curriculum(
            levels=levels,
            initial_level=initial_level,
            window_size=window_size,
            success_threshold=success_threshold,
            reset_history=False,
        )

    ids = _resolve_env_ids(env, env_ids)
    if ids.numel() == 0:
        return env.get_obstacle_curriculum_state()

    # Ignore explicit/manual env.reset() calls. We only want resets caused by terminations.
    if not _is_step_driven_reset(env, ids):
        return env.get_obstacle_curriculum_state()

    history = _ensure_history(env, window_size)

    try:
        success_term = env.termination_manager.get_term(success_term_name)
    except Exception:
        if not getattr(env, "_curriculum_success_term_warning_issued", False):
            print(
                f"[WARN][Curriculum] termination term '{success_term_name}' was not found. "
                "Obstacle curriculum will stay at its current level.",
                flush=True,
            )
            env._curriculum_success_term_warning_issued = True
        return env.get_obstacle_curriculum_state()

    batch_success_mask = success_term[ids].to(dtype=torch.int32)
    batch_total = int(ids.numel())
    batch_success = int(batch_success_mask.sum().item())
    batch_ratio = float(batch_success) / float(batch_total) if batch_total > 0 else 0.0

    history.extend(int(x) for x in batch_success_mask.detach().cpu().tolist())

    decision_count = len(history)
    decision_success = int(sum(history))
    decision_ratio = float(decision_success) / float(decision_count) if decision_count > 0 else 0.0

    promoted = 0.0
    current_level_idx = int(getattr(env, "curriculum_obstacle_level_idx", 0))
    if decision_count >= min_samples and decision_ratio >= success_threshold:
        if current_level_idx < len(levels) - 1:
            current_level_idx += 1
            if hasattr(env, "set_obstacle_curriculum_level"):
                env.set_obstacle_curriculum_level(current_level_idx)
            else:
                env.curriculum_obstacle_level_idx = current_level_idx
                env.curriculum_active_obstacles = int(levels[current_level_idx])

            env.curriculum_promotion_count = int(getattr(env, "curriculum_promotion_count", 0)) + 1
            env.curriculum_last_promotion_step = int(getattr(env, "common_step_counter", 0))
            env.curriculum_last_promotion_ratio = float(decision_ratio)
            promoted = 1.0

            if clear_history_on_promotion:
                history.clear()

    rolling_count = len(history)
    rolling_success = int(sum(history))
    rolling_ratio = float(rolling_success) / float(rolling_count) if rolling_count > 0 else 0.0

    env.curriculum_last_batch_success_ratio = float(batch_ratio)
    env.curriculum_last_batch_success_count = batch_success
    env.curriculum_last_batch_termination_count = batch_total
    env.curriculum_decision_success_ratio = float(decision_ratio)
    env.curriculum_decision_window_size = decision_count
    env.curriculum_recent_success_ratio = float(rolling_ratio)
    env.curriculum_recent_termination_count = rolling_count
    env.curriculum_success_threshold = float(success_threshold)

    return {
        "level_idx": float(getattr(env, "curriculum_obstacle_level_idx", 0)),
        "active_obstacles": float(getattr(env, "curriculum_active_obstacles", 0)),
        "last_batch_success_ratio": float(batch_ratio),
        "last_batch_success_count": float(batch_success),
        "last_batch_termination_count": float(batch_total),
        "decision_success_ratio": float(decision_ratio),
        "decision_window_size": float(decision_count),
        "rolling_success_ratio": float(rolling_ratio),
        "rolling_window_size": float(rolling_count),
        "success_threshold": float(success_threshold),
        "promotion": float(promoted),
        "promotion_count": float(getattr(env, "curriculum_promotion_count", 0)),
    }
