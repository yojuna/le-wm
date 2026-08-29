#!/usr/bin/env python3
"""Unit smoke tests for ReachabilityHead, JEPA hooks, and hindsight split."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reachability import ReachabilityHead  # noqa: E402
from jepa import JEPA  # noqa: E402
from phi_data import (  # noqa: E402
    LiveHindsightPairDataset,
    HindsightPair,
    sample_hindsight_pairs,
    split_episodes,
    usable_episodes,
    build_train_val_datasets,
)


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


def test_goal_cache_id_distinguishes_pixels():
    model = JEPA(
        encoder=nn.Identity(),
        predictor=nn.Identity(),
        action_encoder=nn.Identity(),
    )
    a = torch.zeros(1, 3, 8, 8)
    b = torch.ones(1, 3, 8, 8)
    assert model._goal_cache_id(a) != model._goal_cache_id(b)
    assert model._goal_cache_id(a) == model._goal_cache_id(a.clone())


def test_get_cost_tolerates_goal_cache_key_string():
    """goal_cache_key must not break the goal_* strip / encode path."""

    class _DummyEnc(nn.Module):
        def forward(self, x, interpolate_pos_encoding=True):
            b = x.shape[0]
            return type("O", (), {"last_hidden_state": torch.zeros(b, 1, 192)})()

    class _Act(nn.Module):
        def forward(self, a):
            return torch.zeros(*a.shape[:-1], 192)

    class _Pred(nn.Module):
        def forward(self, emb, act_emb):
            return emb

    model = JEPA(
        encoder=_DummyEnc(),
        predictor=_Pred(),
        action_encoder=_Act(),
        projector=nn.Identity(),
        pred_proj=nn.Identity(),
    )
    # parameter so get_cost can resolve device
    model._probe = nn.Parameter(torch.zeros(1))
    model.reach = ReachabilityHead(input_dim=192, output_dim=64)
    model.plan_cost = "phi_d"
    model.cache_goal_emb = True
    model.clear_goal_cache()
    model._forced_goal_cache_key = "pair:0:seed:0"

    # CEM layout: pixels (B, S, T_hist, C, H, W); candidates (B, S, T_plan, A)
    B, S, hist, plan_t, Hw, act_dim = 1, 4, 3, 5, 8, 2
    info = {
        "pixels": torch.randn(B, S, hist, 3, Hw, Hw),
        "goal": torch.randn(B, S, 1, 3, Hw, Hw),
        "action": torch.randn(B, S, hist, act_dim),
    }
    candidates = torch.randn(B, S, plan_t, act_dim)
    cost = model.get_cost(info, candidates)
    assert cost.shape == (B, S)
    assert model._cached_goal_id == ("key", "pair:0:seed:0")
    # second call hits cache (even if goal pixels change)
    info2 = {
        "pixels": torch.randn(B, S, hist, 3, Hw, Hw),
        "goal": torch.randn(B, S, 1, 3, Hw, Hw),
        "action": torch.randn(B, S, hist, act_dim),
    }
    cost2 = model.get_cost(info2, candidates)
    assert cost2.shape == (B, S)
    # clear_goal_cache must drop forced key too
    model.clear_goal_cache()
    assert model._forced_goal_cache_key is None
    assert model._cached_goal_emb is None



class _FakeEp:
    def __init__(self, L: int, fill: int):
        self.pixels = [
            np.full((16, 16, 3), fill, dtype=np.uint8) for _ in range(L)
        ]

    def __len__(self):
        return len(self.pixels)


class _FakeBank:
    def __init__(self, lengths_fills):
        self.episodes = [_FakeEp(L, f) for L, f in lengths_fills]
        self.num_steps = sum(lengths_fills[i][0] for i in range(len(lengths_fills)))
        self.collector = "weak"
        self.num_success_episodes = 0


def test_getitem_is_indexed_not_random():
    eps = [_FakeEp(40, 1)]
    pairs = [
        HindsightPair(0, 0, 5),
        HindsightPair(0, 3, 2),
    ]
    ds = LiveHindsightPairDataset(eps, pairs)
    a = ds[0]["k"].item()
    b = ds[0]["k"].item()
    assert a == b == 5.0
    assert ds[1]["k"].item() == 2.0


def test_episode_split_disjoint():
    eps = [_FakeEp(40, i) for i in range(10)]
    train, val = split_episodes(eps, val_frac=0.2, seed=0)
    assert len(train) + len(val) == 10
    assert len(val) >= 1 and len(train) >= 1
    assert set(id(e) for e in train).isdisjoint(set(id(e) for e in val))


def test_build_train_val_no_overlap():
    bank = _FakeBank([(50, i) for i in range(12)])
    train_ds, val_ds, meta = build_train_val_datasets(
        bank,
        k_max=25,
        samples_per_epoch=128,
        val_frac=0.25,
        seed=0,
    )
    assert meta["n_train_episodes"] >= 1
    assert meta["n_val_episodes"] >= 1
    assert not meta["train_val_episode_overlap"]
    assert not meta["same_episode_fallback"]
    # same index → same k
    assert train_ds[0]["k"].item() == train_ds[0]["k"].item()


def test_sample_pairs_respects_k_max():
    eps = [_FakeEp(30, 0)]
    pairs = sample_hindsight_pairs(eps, k_max=10, n_samples=200, seed=1)
    assert all(1 <= p.k <= 10 for p in pairs)
    assert all(p.t + p.k < 30 for p in pairs)


if __name__ == "__main__":
    test_shapes_and_zero_self_distance()
    test_grads_only_on_phi()
    test_jepa_criterion_phi_cost()
    test_clear_goal_cache()
    test_goal_cache_id_distinguishes_pixels()
    test_get_cost_tolerates_goal_cache_key_string()
    test_getitem_is_indexed_not_random()
    test_episode_split_disjoint()
    test_build_train_val_no_overlap()
    test_sample_pairs_respects_k_max()
    print("all reachability / phi_data tests passed")
