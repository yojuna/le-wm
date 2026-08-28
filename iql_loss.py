"""Destrade et al. (arXiv:2601.00844) Eq. (1) — IQL expectile value loss.

Paper:

  L_VF = Σ_n Σ_t L_τ²( -1_{s_t ≠ g_n} + γ V_φ̄(s_{t+1}, g_n) - V_φ(s_t, g_n) )

  L_τ²(x) = |τ - 1_{x < 0}| x²

with φ̄ = stop-gradient on the bootstrap value (same network; no target EMA
required by the formula). VF_quasi defaults: γ=0.93, τ=0.60.
"""

from __future__ import annotations

import torch


def expectile_l2(td_error: torch.Tensor, tau: float) -> torch.Tensor:
    """Elementwise L_τ²(x) = |τ - 1_{x<0}| x²."""
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0,1), got {tau}")
    weight = torch.where(
        td_error < 0,
        td_error.new_full((), 1.0 - tau),
        td_error.new_full((), tau),
    )
    return weight * td_error.square()


def iql_vf_loss(
    V_t: torch.Tensor,
    V_tp1: torch.Tensor,
    not_at_goal: torch.Tensor,
    *,
    gamma: float = 0.93,
    tau: float = 0.60,
    reduction: str = "mean",
) -> torch.Tensor:
    """Destrade Eq. (1) expectile TD loss.

    Args:
        V_t: V_φ(s_t, g) — differentiable w.r.t. φ.
        V_tp1: V_φ̄(s_{t+1}, g) — caller must stop-grad (or this will detach).
        not_at_goal: 1_{s_t ≠ g} as float/bool tensor broadcastable to V_t.
        gamma: discount (paper VF_quasi: 0.93).
        tau: expectile (paper VF_quasi: 0.60).
        reduction: "mean" | "sum" | "none".

    Mapping to paper symbols:
        -1_{s≠g}  ↔  -not_at_goal
        γ V_φ̄     ↔  gamma * V_tp1.detach()
        V_φ       ↔  V_t
    """
    not_at_goal = not_at_goal.to(dtype=V_t.dtype)
    # Paper uses stop-grad bootstrap; detach defensively even if caller forgot.
    bootstrap = V_tp1.detach()
    td_error = -not_at_goal + gamma * bootstrap - V_t
    per = expectile_l2(td_error, tau)
    if reduction == "mean":
        return per.mean()
    if reduction == "sum":
        return per.sum()
    if reduction == "none":
        return per
    raise ValueError(f"unknown reduction {reduction!r}")
