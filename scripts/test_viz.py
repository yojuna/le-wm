#!/usr/bin/env python3
"""Guardrails for viz.py (spec 15 v3)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from viz import (  # noqa: E402
    FIG_REGISTRY,
    FigureResult,
    ImaginedFitError,
    RealFittedProjector,
    ca0_fork_from_by_m,
    fig_a1,
    fig_a2,
    fig_d1,
    load_thresholds,
    nn_retrieve,
    nonlinear_id,
    probe_decompose,
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


def test_no_tsne_umap_in_viz_source():
    src = (ROOT / "viz.py").read_text().lower()
    assert "tsne" not in src
    assert "umap" not in src


def test_fig_docstrings_name_scalar():
    seen = set()
    for name, fn in FIG_REGISTRY.items():
        if fn in seen:
            continue
        seen.add(fn)
        doc = inspect.getdoc(fn) or ""
        assert "Motivates:" in doc, f"{name} missing Motivates:"


def test_nn_retrieve_identity():
    bank = np.eye(4)
    idx, dist = nn_retrieve(bank[2], bank)
    assert int(idx) == 2
    assert float(dist) < 1e-8


def _toy_ca0():
    rng = np.random.default_rng(0)
    n, l, d = 4, 10, 8
    z_true = rng.normal(size=(n, l, d)).astype(np.float32)
    z_star = z_true[:, -1].copy()
    z_hat = z_true + 0.3 * rng.normal(size=z_true.shape)
    return {
        "z_true": z_true,
        "z_star": z_star,
        "z_hat": np.stack([z_hat, z_hat], axis=1),
        "m_values": np.array([1, 25], dtype=np.int32),
        "d_end": np.ones((n, 2), dtype=np.float32) * 2.0,
        "d_start": np.full(n, 1.0, dtype=np.float32),
        "toward": np.zeros((n, 2), dtype=bool),
    }


def test_a1_bank_mean_not_example_pair():
    ca0 = _toy_ca0()
    ca0["d_end"][0, 1] = 9.0  # pair 0, m=25
    res = fig_a1(ca0, pair=0)
    bank = float(np.mean(ca0["d_end"][:, 1]))
    assert abs(res.scalars["d_end_open"]["value"] - bank) < 1e-5
    assert abs(res.scalars["example_d_end_open"]["value"] - 9.0) < 1e-5
    assert "example pair" in res.caption["reading_here"].lower()
    import matplotlib.pyplot as plt

    plt.close(res.figure)


def test_a2_has_adjacent_true_z():
    res = fig_a2(_toy_ca0())
    assert "median_adjacent_true_z" in res.scalars
    import matplotlib.pyplot as plt

    plt.close(res.figure)


def test_a5_scalar_is_full_space_bank():
    from viz import fig_a5

    res = fig_a5(_toy_ca0(), pair=0)
    assert res.tier == 2
    assert "mean_angular_error_deg" in res.scalars
    assert "example_plane_angle_deg" in res.scalars
    import matplotlib.pyplot as plt

    plt.close(res.figure)


def test_a3_emits_agent_and_block_shares():
    from viz import fig_a3

    rng = np.random.default_rng(0)
    dump = {
        "z": rng.normal(size=(6, 12, 16)).astype(np.float32),
        "z_hat": rng.normal(size=(6, 12, 16)).astype(np.float32),
        "state": rng.normal(size=(6, 12, 7)).astype(np.float32),
        "factor_names": np.array(
            ["agent_x", "agent_y", "block_x", "block_y", "block_angle", "agent_vx", "agent_vy"]
        ),
        "meta": {"env": "pusht", "collector": "kinematic"},
    }
    dump["state"][:, :, 0] = np.linspace(0, 100, 12)
    res = fig_a3(dump, segment=0)
    assert "agent_xy_share" in res.scalars
    assert "block_xy_share" in res.scalars
    assert res.scalars["bank"]["value"] == "pusht/kinematic"
    import matplotlib.pyplot as plt

    plt.close(res.figure)


def test_a1_returns_figure_result_with_in_plane():
    res = fig_a1(_toy_ca0(), pair=0)
    assert isinstance(res, FigureResult)
    assert res.tier == 1
    assert "in_plane_drift_fraction" in res.scalars
    assert res.figure._viz_captured_variance > 0
    assert "what" in res.caption and "would_overturn" in res.caption
    import matplotlib.pyplot as plt

    plt.close(res.figure)


def test_a2_fork_and_onestep():
    ca0 = _toy_ca0()
    res = fig_a2(ca0)
    assert isinstance(res, FigureResult)
    assert "median_onestep_error" in res.scalars
    import matplotlib.pyplot as plt

    plt.close(res.figure)


def test_thresholds_match_ca0_cuts():
    th = load_thresholds()["ca0"]
    assert th["m1_toward_guard"] == 0.9
    assert th["m1_d_end_guard"] == 1.0


def test_fork_m1_guard():
    by = {
        1: {"frac_toward": 0.84, "mean_d_end": 1.43},
        5: {"frac_toward": 0.62, "mean_d_end": 2.35},
    }
    out = ca0_fork_from_by_m(by)
    assert out["fork"] == "CA0-INFIDELITY"


def test_probe_decompose_identity():
    v = np.array([1.0, 0.0, 0.0])
    shares = probe_decompose(v, {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0])})
    assert shares["x"] > 0.99
    assert shares["y"] < 0.01


def test_nonlinear_id_gaussian_plane():
    rng = np.random.default_rng(0)
    xy = rng.normal(size=(80, 2))
    z = np.concatenate([xy, np.zeros((80, 6))], axis=1)
    nid = nonlinear_id(z, max_points=80)
    assert 1.0 < nid < 5.0


def test_d1_has_nonlinear_id():
    rng = np.random.default_rng(0)
    dump = {
        "z": rng.normal(size=(8, 12, 16)).astype(np.float32),
        "state": rng.normal(size=(8, 12, 7)).astype(np.float32),
        "factor_names": np.array(
            ["agent_x", "agent_y", "block_x", "block_y", "block_angle", "agent_vx", "agent_vy"]
        ),
        "meta": {"env": "pusht"},
    }
    res = fig_d1(dump)
    assert "nonlinear_id" in res.scalars
    assert "elbow_90" in res.scalars
    import matplotlib.pyplot as plt

    plt.close(res.figure)


if __name__ == "__main__":
    test_refuses_imagined_fit()
    test_captured_variance_on_2d_gaussian()
    test_no_tsne_umap_in_viz_source()
    test_fig_docstrings_name_scalar()
    test_nn_retrieve_identity()
    test_a1_returns_figure_result_with_in_plane()
    test_a1_bank_mean_not_example_pair()
    test_a2_fork_and_onestep()
    test_a2_has_adjacent_true_z()
    test_a5_scalar_is_full_space_bank()
    test_a3_emits_agent_and_block_shares()
    test_thresholds_match_ca0_cuts()
    test_fork_m1_guard()
    test_probe_decompose_identity()
    test_nonlinear_id_gaussian_plane()
    test_d1_has_nonlinear_id()
    print("all viz tests passed")
