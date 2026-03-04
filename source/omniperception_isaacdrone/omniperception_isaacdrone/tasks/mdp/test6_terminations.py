# omniperception_isaacdrone/tasks/mdp/test6_terminations.py

from __future__ import annotations

import torch
import isaaclab.envs.mdp as mdp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def termination_reached_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)
    goal = getattr(env, "goal_pos_w", None)
    if goal is None:
        return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)

    if goal.device != pos.device:
        goal = goal.to(pos.device)

    if goal.shape[0] == 1 and pos.shape[0] > 1:
        goal = goal.expand(pos.shape[0], 3)
    elif goal.shape[0] != pos.shape[0]:
        goal = goal[:1].expand(pos.shape[0], 3)

    dist = torch.norm(goal - pos, dim=-1)
    return dist < float(threshold)


def termination_out_of_workspace(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    x_bounds: tuple[float, float] = (-60.0, 60.0),
    y_bounds: tuple[float, float] = (-60.0, 60.0),
    z_bounds: tuple[float, float] = (0.0, 10.0),
) -> torch.Tensor:
    pos = mdp.root_pos_w(env, asset_cfg=asset_cfg)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    z_min, z_max = float(z_bounds[0]), float(z_bounds[1])

    return (
        (pos[:, 0] < x_min)
        | (pos[:, 0] > x_max)
        | (pos[:, 1] < y_min)
        | (pos[:, 1] > y_max)
        | (pos[:, 2] < z_min)
        | (pos[:, 2] > z_max)
    )


def termination_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Collision termination using ContactSensor.

    注意：
      - 如果 ContactSensorCfg.filter_prim_paths_expr=None，则 force_matrix 通常不可用，
        这时退化使用 net_forces_w（更通用）。
    """
    out = torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)

    # get sensor
    try:
        contact_sensor = env.scene[sensor_cfg.name]
    except Exception:
        return out

    data = contact_sensor.data

    # body_ids robustness
    body_ids = getattr(sensor_cfg, "body_ids", None)
    if body_ids is None:
        body_ids = slice(None)

    # try force matrix first
    fm_hist = getattr(data, "force_matrix_w_history", None)
    fm = None
    if isinstance(fm_hist, torch.Tensor) and fm_hist.dim() == 5 and fm_hist.shape[1] > 0:
        fm = fm_hist[:, 0]  # (N,B,M,3)

    if fm is None:
        fm_now = getattr(data, "force_matrix_w", None)
        if isinstance(fm_now, torch.Tensor) and fm_now.dim() == 4:
            fm = fm_now

    if fm is not None:
        fm_sel = fm[:, body_ids, :, :]
        f_norm = torch.norm(fm_sel, dim=-1)
        f_norm = torch.nan_to_num(f_norm, nan=0.0, posinf=0.0, neginf=0.0)
        max_force = f_norm.reshape(f_norm.shape[0], -1).max(dim=-1).values
        return max_force > float(threshold)

    # fallback: net forces
    nf_hist = getattr(data, "net_forces_w_history", None)
    nf = None
    if isinstance(nf_hist, torch.Tensor) and nf_hist.dim() == 4 and nf_hist.shape[1] > 0:
        nf = nf_hist[:, 0]
    if nf is None:
        nf_now = getattr(data, "net_forces_w", None)
        if isinstance(nf_now, torch.Tensor) and nf_now.dim() == 3:
            nf = nf_now
    if nf is None:
        return out

    nf_sel = nf[:, body_ids, :]
    f_norm = torch.norm(nf_sel, dim=-1)
    max_force = f_norm.reshape(f_norm.shape[0], -1).max(dim=-1).values
    return max_force > float(threshold)
