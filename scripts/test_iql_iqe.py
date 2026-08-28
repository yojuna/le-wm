#!/usr/bin/env python3
"""Property tests for IQE-sum and Destrade Eq. (1) IQL expectile loss."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iqe import iqe_sum, reshape_phi  # noqa: E402


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


def main():
    test_iqe_zero_self_distance()
    test_iqe_nonnegative()
    test_iqe_asymmetry_exists()
    test_iqe_reshape_roundtrip()
    test_iqe_homogeneity_positive_scale()
    print("iqe tests OK")


if __name__ == "__main__":
    main()
