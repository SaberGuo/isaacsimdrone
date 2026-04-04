"""Dijkstra-based navigation utilities for geodesic distance computation.

This module implements fast geodesic distance computation using GPU-accelerated
algorithms for reward shaping in drone navigation tasks.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


class DijkstraNavigator:
    """Computes geodesic distances using fast GPU-accelerated algorithms.

    The geodesic distance represents the shortest path length from any position
to the goal, accounting for obstacles. This provides better reward shaping
    than Euclidean distance in cluttered environments.
    """

    def __init__(
        self,
        grid_size: int = 160,
        cell_size: float = 1.0,
        workspace_origin: Tuple[float, float] = (-80.0, -80.0),
        max_distance: float = 300.0,
    ):
        """Initialize the navigator.

        Args:
            grid_size: Number of cells per grid dimension (grid_size x grid_size).
            cell_size: Size of each cell in meters.
            workspace_origin: (x_min, y_min) of the workspace in world coordinates.
            max_distance: Maximum distance value for unreachable cells.
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.workspace_origin = workspace_origin
        self.max_distance = max_distance

    def world_to_grid(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert world coordinates to grid cell indices.

        Args:
            positions: (N, 2) or (N, 3) tensor of world positions (x, y, [z]).

        Returns:
            Tuple of (grid_x, grid_y) indices, each (N,) tensor.
        """
        x = positions[:, 0]
        y = positions[:, 1]

        grid_x = ((x - self.workspace_origin[0]) / self.cell_size).long()
        grid_y = ((y - self.workspace_origin[1]) / self.cell_size).long()

        # Clamp to valid grid range
        grid_x = torch.clamp(grid_x, 0, self.grid_size - 1)
        grid_y = torch.clamp(grid_y, 0, self.grid_size - 1)

        return grid_x, grid_y

    def grid_to_world(self, grid_x: torch.Tensor, grid_y: torch.Tensor) -> torch.Tensor:
        """Convert grid cell indices to world coordinates (cell centers).

        Args:
            grid_x: (N,) tensor of x indices.
            grid_y: (N,) tensor of y indices.

        Returns:
            (N, 2) tensor of world (x, y) coordinates.
        """
        x = grid_x.float() * self.cell_size + self.workspace_origin[0] + self.cell_size / 2
        y = grid_y.float() * self.cell_size + self.workspace_origin[1] + self.cell_size / 2

        return torch.stack([x, y], dim=-1)

    def build_occupancy_grid_from_obstacles(
        self,
        obstacle_positions: torch.Tensor,
        obstacle_size: float = 1.5,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Build 2D occupancy grid from obstacle positions.

        Args:
            obstacle_positions: (N_obstacles, 3) tensor of obstacle positions.
            obstacle_size: Size to inflate each obstacle (meters).
            device: Device for the output tensor.

        Returns:
            (grid_size, grid_size) boolean tensor, True = occupied.
        """
        grid = torch.zeros(
            (self.grid_size, self.grid_size), dtype=torch.bool, device=device
        )

        if obstacle_positions.shape[0] == 0:
            return grid

        # Convert obstacle positions to grid coordinates
        grid_x, grid_y = self.world_to_grid(obstacle_positions)

        # Inflate obstacles by obstacle_size meters
        inflate_cells = max(1, int(torch.ceil(torch.tensor(obstacle_size / self.cell_size)).item()))

        # Vectorized obstacle marking
        for dx in range(-inflate_cells, inflate_cells + 1):
            for dy in range(-inflate_cells, inflate_cells + 1):
                # Skip corners for circular inflation
                if dx * dx + dy * dy > inflate_cells * inflate_cells:
                    continue
                nx = torch.clamp(grid_x + dx, 0, self.grid_size - 1)
                ny = torch.clamp(grid_y + dy, 0, self.grid_size - 1)
                grid[ny, nx] = True

        return grid

    def compute_distance_field_fast(
        self,
        occupancy_grid: torch.Tensor,
        goal_pos: torch.Tensor,
        max_iterations: int = 1000,
    ) -> torch.Tensor:
        """Compute geodesic distance field using fast GPU-accelerated wave propagation.

        This uses an iterative wave propagation algorithm that is much faster than
        Dijkstra for GPU-parallel computation.

        Args:
            occupancy_grid: (grid_size, grid_size) boolean tensor, True = occupied.
            goal_pos: (2,) or (3,) tensor of goal position in world coordinates.
            max_iterations: Maximum propagation iterations (safety limit).

        Returns:
            (grid_size, grid_size) tensor of distances from each cell to goal.
        """
        device = occupancy_grid.device

        # Convert goal to grid coordinates
        if goal_pos.dim() == 1:
            goal_pos = goal_pos.unsqueeze(0)
        goal_grid_x, goal_grid_y = self.world_to_grid(goal_pos)
        goal_gx, goal_gy = goal_grid_x[0].item(), goal_grid_y[0].item()

        # Initialize distance field
        distance_field = torch.full(
            (self.grid_size, self.grid_size),
            self.max_distance,
            dtype=torch.float32,
            device=device,
        )

        # Check if goal is inside obstacle - find nearest free cell
        if occupancy_grid[goal_gy, goal_gx]:
            found = False
            search_radius = 1
            while not found and search_radius < self.grid_size // 2:
                # Search in expanding squares
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        if abs(dx) != search_radius and abs(dy) != search_radius:
                            continue
                        ny, nx = goal_gy + dy, goal_gx + dx
                        if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                            if not occupancy_grid[ny, nx]:
                                goal_gy, goal_gx = ny, nx
                                found = True
                                break
                    if found:
                        break
                search_radius += 1

        # Set goal distance to 0
        distance_field[goal_gy, goal_gx] = 0.0

        # Fast wave propagation using convolution (no grad to save memory)
        with torch.no_grad():
            current = distance_field.clone()
            mask = ~occupancy_grid  # Free space mask

            # Limit iterations to prevent hanging (grid diagonal is max needed)
            max_needed_iterations = int(self.grid_size * 1.5)  # Diagonal + margin
            num_iterations = min(max_iterations, max_needed_iterations)

            for iter_idx in range(num_iterations):
                # Pad for convolution
                padded = F.pad(current.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='constant', value=self.max_distance)

                # Compute minimum neighbor distance + step cost
                # Use inplace operations to reduce memory allocation
                n1 = padded[:, :, :-2, :-2] + 1.414 * self.cell_size  # top-left
                n2 = padded[:, :, :-2, 1:-1] + 1.0 * self.cell_size   # top
                n3 = padded[:, :, :-2, 2:] + 1.414 * self.cell_size   # top-right
                n4 = padded[:, :, 1:-1, :-2] + 1.0 * self.cell_size   # left
                n5 = padded[:, :, 1:-1, 2:] + 1.0 * self.cell_size    # right
                n6 = padded[:, :, 2:, :-2] + 1.414 * self.cell_size   # bottom-left
                n7 = padded[:, :, 2:, 1:-1] + 1.0 * self.cell_size    # bottom
                n8 = padded[:, :, 2:, 2:] + 1.414 * self.cell_size    # bottom-right

                # Stack and find min without keeping intermediate tensors
                new_distances = torch.min(torch.stack([n1, n2, n3, n4, n5, n6, n7, n8], dim=1), dim=1)[0]

                # Clean up intermediate tensors
                del n1, n2, n3, n4, n5, n6, n7, n8, padded

                # Apply mask and keep goal at 0
                new_distances = torch.where(mask, new_distances, self.max_distance)
                new_distances[goal_gy, goal_gx] = 0.0

                # Check convergence
                diff = (current - new_distances).abs().max()
                current = new_distances

                if diff < 0.01:
                    break

            return current

    def compute_distance_field(
        self,
        occupancy_grid: torch.Tensor,
        goal_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geodesic distance field (wrapper for fast implementation).

        Args:
            occupancy_grid: (grid_size, grid_size) boolean tensor, True = occupied.
            goal_pos: (2,) or (3,) tensor of goal position in world coordinates.

        Returns:
            (grid_size, grid_size) tensor of distances from each cell to goal.
        """
        return self.compute_distance_field_fast(occupancy_grid, goal_pos)

    def batch_compute_distance_fields(
        self,
        occupancy_grids: torch.Tensor,
        goal_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distance fields for multiple environments.

        Args:
            occupancy_grids: (N, grid_size, grid_size) boolean tensor.
            goal_positions: (N, 2) or (N, 3) tensor of goal positions.

        Returns:
            (N, grid_size, grid_size) tensor of distance fields.
        """
        N = occupancy_grids.shape[0]
        device = occupancy_grids.device

        distance_fields = torch.full(
            (N, self.grid_size, self.grid_size),
            self.max_distance,
            dtype=torch.float32,
            device=device,
        )

        for i in range(N):
            distance_fields[i] = self.compute_distance_field(
                occupancy_grids[i], goal_positions[i]
            )

        return distance_fields

    def get_geodesic_distance(
        self,
        positions: torch.Tensor,
        distance_field: torch.Tensor,
    ) -> torch.Tensor:
        """Get geodesic distance at specified positions.

        Args:
            positions: (N, 2) or (N, 3) tensor of world positions.
            distance_field: (grid_size, grid_size) or (N, grid_size, grid_size) tensor.

        Returns:
            (N,) tensor of geodesic distances.
        """
        grid_x, grid_y = self.world_to_grid(positions)

        if distance_field.dim() == 2:
            # Single distance field for all positions
            distances = distance_field[grid_y, grid_x]
        else:
            # Per-environment distance fields
            N = positions.shape[0]
            distances = torch.zeros(N, dtype=torch.float32, device=positions.device)
            for i in range(N):
                distances[i] = distance_field[i, grid_y[i], grid_x[i]]

        return distances


def batched_dijkstra_reward(
    positions: torch.Tensor,
    prev_positions: torch.Tensor,
    goal_positions: torch.Tensor,
    distance_fields: torch.Tensor,
    navigator: DijkstraNavigator,
    speed_ref: float = 4.0,
    dt: float = 1.0 / 60.0,
    clip: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Dijkstra-based progress reward for a batch of agents.

    Args:
        positions: (N, 3) current positions.
        prev_positions: (N, 3) previous positions.
        goal_positions: (N, 3) goal positions.
        distance_fields: (N, H, W) precomputed distance fields.
        navigator: DijkstraNavigator instance.
        speed_ref: Reference speed for normalization.
        dt: Time step.
        clip: Reward clipping value.

    Returns:
        Tuple of (reward, current_distance, prev_distance).
    """
    # Get geodesic distances
    d_current = navigator.get_geodesic_distance(positions, distance_fields)
    d_prev = navigator.get_geodesic_distance(prev_positions, distance_fields)

    # Compute progress reward (increase in distance = negative reward)
    delta = d_prev - d_current
    towards_speed = delta / max(dt, 1e-6)

    reward = torch.clamp(towards_speed / speed_ref, -clip, clip)

    return reward, d_current, d_prev
