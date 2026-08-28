"""Thin reachability projection φ for lewm-phi planning costs."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ReachProjection(nn.Module):
    """Map JEPA embedding z -> u (thin MLP)."""

    def __init__(self, input_dim: int = 192, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ReachabilityHead(nn.Module):
    """φ + Euclidean distance used as CEM planning cost / hindsight target."""

    def __init__(
        self,
        input_dim: int = 192,
        hidden_dim: int = 256,
        output_dim: int = 64,
    ):
        super().__init__()
        self.phi = ReachProjection(input_dim, hidden_dim, output_dim)

    def project(self, z: torch.Tensor, *, detach_z: bool = True) -> torch.Tensor:
        if detach_z:
            z = z.detach()
        # Flatten leading dims except feature
        flat = z.reshape(-1, z.size(-1))
        u = self.phi(flat)
        return u.view(*z.shape[:-1], u.size(-1))

    def pairwise_distance(
        self, z_t: torch.Tensor, z_tk: torch.Tensor, *, detach_z: bool = True
    ) -> torch.Tensor:
        """Return Euclidean ‖φ(z_t) - φ(z_tk)‖ with shape broadcast of leading dims."""
        u_t = self.project(z_t, detach_z=detach_z)
        u_tk = self.project(z_tk, detach_z=detach_z)
        return torch.linalg.vector_norm(u_t - u_tk, ord=2, dim=-1)

    def distance(
        self, z: torch.Tensor, z_star: torch.Tensor, *, detach_z: bool = True
    ) -> torch.Tensor:
        """Alias for pairwise_distance (planning API)."""
        return self.pairwise_distance(z, z_star, detach_z=detach_z)

    def planning_cost(
        self, pred_emb: torch.Tensor, goal_emb: torch.Tensor
    ) -> torch.Tensor:
        """CEM cost from last predicted / goal frames.

        pred_emb: (B, S, T, D) or (B, S, 1, D)
        goal_emb: (B, S, T, D) or (B, S, 1, D)
        returns: (B, S)
        """
        pred_last = pred_emb[..., -1, :]
        goal_last = goal_emb[..., -1, :]
        # Expand goal if needed
        while goal_last.ndim < pred_last.ndim:
            goal_last = goal_last.unsqueeze(-2)
        if goal_last.shape[-2] == 1 and pred_last.shape[-2] != 1:
            goal_last = goal_last.expand_as(pred_last)
        d = self.pairwise_distance(pred_last, goal_last, detach_z=True)
        # If still have a time dim, take last
        if d.ndim > 2:
            d = d[..., -1]
        return d
