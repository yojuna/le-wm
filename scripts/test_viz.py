#!/usr/bin/env python3
"""Guardrails for viz.py (spec 15 §4)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz import (  # noqa: E402
    FIG_REGISTRY,
    ImaginedFitError,
    RealFittedProjector,
    fig_oracle_overlay,
    fig_rank_spectrum,
    nn_retrieve,
)


def test_refuses_imagined_fit():
    p = RealFittedProjector()
    z = np.random.randn(20, 8)
    try:
        p.fit(z, imagined=True)
    except ImaginedFitError:
        return
    raise AssertionError("expected ImaginedFitError")


def test_captured_variance_on_2d_gaussian():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 1))
    z = np.concatenate([x, 0.01 * rng.normal(size=(200, 1))], axis=1)
    p = RealFittedProjector(n_components=2)
    p.fit(z)
    assert p.captured_variance > 0.99
    y = p.transform(z)
    assert y.shape == (200, 2)


def test_transform_imagined_after_real_fit():
    rng = np.random.default_rng(1)
    real = rng.normal(size=(50, 6))
    hat = real + 0.5 * rng.normal(size=(50, 6))
    p = RealFittedProjector()
    p.fit(real)
    t = p.transform(hat)
    assert t.shape == (50, 2)


def test_no_tsne_umap_in_viz_source():
    src = (ROOT / "viz.py").read_text().lower()
    assert "tsne" not in src
    assert "umap" not in src


def test_fig_docstrings_name_scalar():
    for name, fn in FIG_REGISTRY.items():
        doc = inspect.getdoc(fn) or ""
        assert "Motivates:" in doc, f"{name} missing Motivates: in docstring"


def test_nn_retrieve_identity():
    bank = np.eye(4)
    idx, dist = nn_retrieve(bank[2], bank)
    assert int(idx) == 2
    assert float(dist) < 1e-8


def test_oracle_overlay_attaches_variance(tmp_path: Path | None = None):
    rng = np.random.default_rng(0)
    n, l, d = 3, 10, 8
    z_true = rng.normal(size=(n, l, d)).astype(np.float32)
    z_star = z_true[:, -1].copy()
    z_hat = z_true + 0.2 * rng.normal(size=z_true.shape)
    ca0 = {
        "z_true": z_true,
        "z_star": z_star,
        "z_hat": np.stack([z_hat, z_hat], axis=1),
        "m_values": np.array([5, 25], dtype=np.int32),
        "d_end": np.ones((n, 2), dtype=np.float32),
        "d_start": np.full(n, 2.0, dtype=np.float32),
        "toward": np.zeros((n, 2), dtype=bool),
    }
    fig = fig_oracle_overlay(ca0, pair=0)
    assert fig._viz_captured_variance > 0
    assert "CA0" in fig._viz_motivates
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_rank_spectrum_on_fake_dump():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(8, 12, 16)).astype(np.float32)
    state = rng.normal(size=(8, 12, 7)).astype(np.float32)
    dump = {
        "z": z,
        "state": state,
        "factor_names": np.array(
            ["agent_x", "agent_y", "block_x", "block_y", "block_angle", "agent_vx", "agent_vy"]
        ),
        "meta": {"env": "pusht"},
    }
    fig = fig_rank_spectrum(dump)
    assert "elbow" in fig._viz_scalars or "elbow_90" in fig._viz_scalars
    import matplotlib.pyplot as plt

    plt.close(fig)


if __name__ == "__main__":
    test_refuses_imagined_fit()
    test_captured_variance_on_2d_gaussian()
    test_transform_imagined_after_real_fit()
    test_no_tsne_umap_in_viz_source()
    test_fig_docstrings_name_scalar()
    test_nn_retrieve_identity()
    test_oracle_overlay_attaches_variance()
    test_rank_spectrum_on_fake_dump()
    print("all viz tests passed")
