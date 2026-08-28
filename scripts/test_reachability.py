#!/usr/bin/env python3
"""Unit smoke tests for ReachabilityHead + JEPA plan_cost hook."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reachability import ReachabilityHead  # noqa: E402
from jepa import JEPA  # noqa: E402


def test_shapes_and_zero_self_distance():
    head = ReachabilityHead(input_dim=192, output_dim=64)
    z = torch.randn(4, 192)
    d = head.distance(z, z)
    assert d.shape == (4,)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)


def test_grads_only_on_phi():
    head = ReachabilityHead(input_dim=192, output_dim=64)
    z = torch.randn(8, 192, requires_grad=True)
    z_star = torch.randn(8, 192, requires_grad=True)
    loss = head.pairwise_distance(z, z_star, detach_z=True).mean()
    loss.backward()
    assert z.grad is None
    assert z_star.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())


def test_jepa_criterion_phi_cost():
    class _DummyEnc(nn.Module):
        def forward(self, x, interpolate_pos_encoding=True):
            b = x.shape[0]
            return type("O", (), {"last_hidden_state": torch.zeros(b, 1, 192)})()

    model = JEPA(
        encoder=_DummyEnc(),
        predictor=nn.Identity(),
        action_encoder=nn.Identity(),
    )
    model.reach = ReachabilityHead(input_dim=192, output_dim=64)
    model.plan_cost = "phi_d"

    B, S, T, D = 2, 5, 4, 192
    info = {
        "predicted_emb": torch.randn(B, S, T, D),
        "goal_emb": torch.randn(B, S, 1, D),
    }
    cost = model.criterion(info)
    assert cost.shape == (B, S)
    assert torch.isfinite(cost).all()


def test_clear_goal_cache():
    class _DummyEnc(nn.Module):
        def forward(self, x, interpolate_pos_encoding=True):
            b = x.shape[0]
            return type("O", (), {"last_hidden_state": torch.zeros(b, 1, 192)})()

    model = JEPA(
        encoder=_DummyEnc(),
        predictor=nn.Identity(),
        action_encoder=nn.Identity(),
    )
    model._cached_goal_emb = torch.zeros(1, 1, 192)
    model._cached_goal_id = "x"
    model.clear_goal_cache()
    assert model._cached_goal_emb is None
    assert model._cached_goal_id is None


if __name__ == "__main__":
    test_shapes_and_zero_self_distance()
    test_grads_only_on_phi()
    test_jepa_criterion_phi_cost()
    test_clear_goal_cache()
    print("all reachability tests passed")
