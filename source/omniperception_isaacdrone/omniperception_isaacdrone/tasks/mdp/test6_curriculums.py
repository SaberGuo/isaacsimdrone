from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def update_obstacle_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice | torch.Tensor,
    levels: Sequence[int] = (0, 10, 20, 40, 60, 100),
    success_term_name: str = "reached_goal",
    success_threshold: float = 0.8,
    window_size: int = 200,
) -> dict[str, float]:
    """
    根据最近 window_size 次环境 reset 时的成功率，动态更新当前障碍物等级。
    返回的字典会被 IsaacLab 的 CurriculumManager 自动捕获，并加上 "Curriculum/" 前缀写入 infos 中，
    随后由训练脚本推送到 TensorBoard。
    """
    # 1. 初始化环境的状态变量
    if not hasattr(env.unwrapped, "obstacle_level_idx"):
        env.unwrapped.obstacle_level_idx = 0
        env.unwrapped.obstacle_history = deque(maxlen=window_size)
        env.unwrapped.current_obstacle_count = levels[0]
        # 强制使得在第一次训练开始时能够刷一次障碍物
        env.unwrapped.obstacle_level_changed = True

    # 2. 计算当前的成功率（如果历史队列为空则为 0.0）
    success_rate = 0.0
    if len(env.unwrapped.obstacle_history) > 0:
        success_rate = sum(env.unwrapped.obstacle_history) / len(env.unwrapped.obstacle_history)

    # 如果本回合没有任何环境触发重置，直接返回当前统计数据供 TensorBoard 记录
    if env_ids is None or (isinstance(env_ids, torch.Tensor) and len(env_ids) == 0):
        return {
            "active_count": float(env.unwrapped.current_obstacle_count),
            "level_idx": float(env.unwrapped.obstacle_level_idx),
            "success_rate": float(success_rate),
            "success_threshold": float(success_threshold),
        }

    # 3. 获取当前步触发重置的环境的成功信号
    success_term = env.termination_manager.get_term(success_term_name)
    if success_term is not None:
        # 提取当前重置环境的 bool 状态 (1.0 代表成功，0.0 代表失败/超时等)
        successes = success_term[env_ids].float()
        env.unwrapped.obstacle_history.extend(successes.cpu().tolist())
        
        # 重新计算更新后的成功率
        success_rate = sum(env.unwrapped.obstacle_history) / len(env.unwrapped.obstacle_history)

    # 4. 检查是否满足晋级条件
    if len(env.unwrapped.obstacle_history) >= window_size:
        if success_rate >= success_threshold and env.unwrapped.obstacle_level_idx < len(levels) - 1:
            env.unwrapped.obstacle_level_idx += 1
            env.unwrapped.current_obstacle_count = levels[env.unwrapped.obstacle_level_idx]
            env.unwrapped.obstacle_history.clear()  # 晋级后清空历史，重新计算下一阶段的成功率
            env.unwrapped.obstacle_level_changed = True
            
            print(f"[CURRICULUM] 成功率 {success_rate:.2f} >= {success_threshold} | 晋级到难度等级: {env.unwrapped.obstacle_level_idx} | 障碍物数量: {env.unwrapped.current_obstacle_count}")
            success_rate = 0.0  # 晋级后由于历史已清空，成功率归零以供返回

    # 5. 返回所有需要可视化的关键指标
    return {
        "active_count": float(env.unwrapped.current_obstacle_count),
        "level_idx": float(env.unwrapped.obstacle_level_idx),
        "success_rate": float(success_rate),
        "success_threshold": float(success_threshold),
    }
