#!/usr/bin/env python3
"""Property tests for IQE-sum and Destrade Eq. (1) IQL expectile loss."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iqe import iqe_sum, reshape_phi  # noqa: E402
from iql_loss import expectile_l2, iql_vf_loss  # noqa: E402


def test_iqe_zero_self_distance():
    torch.manual_seed(0)
    phi = torch.randn(16, 64)
    d = iqe_sum(phi, phi, k=8, l=8)
    assert d.shape == (16,)
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-5)


def test_iqe_nonnegative():
    torch.manual_seed(1)
    a = torch.randn(32, 64)
    b = torch.randn(32, 64)
    d = iqe_sum(a, b, k=8, l=8)
    assert (d >= -1e-6).all()
    assert torch.isfinite(d).all()


def test_iqe_asymmetry_exists():
    torch.manual_seed(2)
    # Construct a pair where one-sided intervals differ.
    u = torch.zeros(1, 8, 8)
    v = torch.zeros(1, 8, 8)
    u[0, 0, 0] = 0.0
    v[0, 0, 0] = 1.0
    d_uv = iqe_sum(u, v)
    d_vu = iqe_sum(v, u)
    assert float(d_uv) > 0.5
    assert float(d_vu) < 1e-5  # intervals [1, max(1,0)] = [1,1] measure 0
    assert not torch.allclose(d_uv, d_vu)


def test_iqe_reshape_roundtrip():
    phi = torch.randn(4, 64)
    u = reshape_phi(phi, k=8, l=8)
    assert u.shape == (4, 8, 8)
    d_flat = iqe_sum(phi, phi * 0 + 1, k=8, l=8)
    d_shaped = iqe_sum(u, torch.ones_like(u))
    assert torch.allclose(d_flat, d_shaped, atol=1e-5)


def test_iqe_homogeneity_positive_scale():
    """IQE components are positive-homogeneous in interval lengths."""
    torch.manual_seed(3)
    u = torch.randn(8, 8, 8)
    v = torch.randn(8, 8, 8)
    d1 = iqe_sum(u, v)
    d2 = iqe_sum(2 * u, 2 * v)
    assert torch.allclose(d2, 2 * d1, atol=1e-4)


def test_expectile_weights():
    tau = 0.60
    pos = torch.tensor([2.0])
    neg = torch.tensor([-2.0])
    # x>0 → weight τ; x<0 → weight 1-τ
    assert torch.allclose(expectile_l2(pos, tau), torch.tensor([tau * 4.0]))
    assert torch.allclose(expectile_l2(neg, tau), torch.tensor([(1 - tau) * 4.0]))


def test_iql_s_equals_g_zero_reward_term():
    """When s==g, not_at_goal=0 so immediate term vanishes."""
    V_t = torch.tensor([0.0], requires_grad=True)
    V_tp1 = torch.tensor([-1.0])  # unused if gamma*V cancels in check
    loss = iql_vf_loss(
        V_t,
        V_tp1,
        not_at_goal=torch.tensor([0.0]),
        gamma=0.93,
        tau=0.60,
        reduction="none",
    )
    # td = 0 + 0.93*(-1) - 0 = -0.93
    expected = expectile_l2(torch.tensor([-0.93]), 0.60)
    assert torch.allclose(loss, expected, atol=1e-5)


def test_iql_bootstrap_stopgrad():
    V_t = torch.tensor([1.0], requires_grad=True)
    V_tp1 = torch.tensor([2.0], requires_grad=True)
    loss = iql_vf_loss(
        V_t,
        V_tp1,
        not_at_goal=torch.tensor([1.0]),
        gamma=0.93,
        tau=0.60,
    )
    loss.backward()
    assert V_t.grad is not None and float(V_t.grad.abs()) > 0
    assert V_tp1.grad is None  # stop-grad on bootstrap


def test_iql_shapes():
    B = 7
    V_t = torch.randn(B, requires_grad=True)
    V_tp1 = torch.randn(B)
    mask = torch.ones(B)
    loss = iql_vf_loss(V_t, V_tp1, mask, gamma=0.93, tau=0.60)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def main():
    test_iqe_zero_self_distance()
    test_iqe_nonnegative()
    test_iqe_asymmetry_exists()
    test_iqe_reshape_roundtrip()
    test_iqe_homogeneity_positive_scale()
    test_expectile_weights()
    test_iql_s_equals_g_zero_reward_term()
    test_iql_bootstrap_stopgrad()
    test_iql_shapes()
    print("iqe + iql_loss tests OK")


if __name__ == "__main__":
    main()
