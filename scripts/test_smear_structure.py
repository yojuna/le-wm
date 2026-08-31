#!/usr/bin/env python3
"""CPU tests for smear_structure cuts and subspaces."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from smear_structure import (  # noqa: E402
    analyze_bank,
    energy_share,
    live_dead_masks,
    pca_basis,
)


TH = {
    "occupancy_dead_rel": 0.001,
    "occupancy_live_var_frac": 0.90,
    "motion_var_frac": 0.90,
    "knn_k": 15,
    "knn_exclude_dt": 2,
    "neighbor_cosine_systematic_at_or_above": 0.30,
    "neighbor_cosine_random_below": 0.10,
    "energy_majority": 0.50,
    "pose_from_perp_r2_physics_at_or_above": 0.20,
    "pose_from_perp_r2_junk_below": 0.05,
}


def test_dead_mask_matches_d1_rel_cut():
    eig = np.array([1.0, 0.5, 1e-6, 1e-9], dtype=np.float64)
    m = live_dead_masks(eig, dead_rel=0.001, live_var_frac=0.90)
    assert m["dead"].tolist() == [False, False, True, True]
    assert m["k90"] >= 1


def test_energy_share_on_axis():
    basis = np.eye(4)
    vec = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0]])
    mask = np.array([True, False, False, False])
    assert abs(energy_share(vec, basis, mask) - 0.5) < 1e-9  # median of {1, 0}


def _toy_ca0(*, leak_dead: bool, mix_motion: bool):
    rng = np.random.default_rng(0)
    n, L, d = 12, 8, 16
    # occupancy: only first 4 dims live
    z = np.zeros((n, L, d))
    z[..., :4] = rng.normal(size=(n, L, 4))
    true_d = np.diff(z, axis=1)
    hat = z.copy()
    if leak_dead:
        hat[:, 1:, 12] = z[:, :-1, 12] + 3.0  # spray into a dead dim
        hat[:, 1:, :4] = z[:, 1:, :4]
    elif mix_motion:
        # add someone else's true Δz (in live motion span)
        rolled = np.roll(true_d, 1, axis=0)
        hat[:, 1:] = z[:, :-1] + true_d + rolled
    else:
        hat[:, 1:] = z[:, 1:]
    st = np.zeros((n, L, 7))
    st[..., :4] = z[..., :4]
    return {
        "z_true": z.astype(np.float32),
        "z_hat": hat[:, None].astype(np.float32),
        "m_values": np.array([1], dtype=np.int32),
        "path_state": st.astype(np.float32),
    }


def test_dead_leak_decision():
    ca0 = _toy_ca0(leak_dead=True, mix_motion=False)
    actions = np.zeros((12, 7, 2), dtype=np.float32)
    out = analyze_bank(ca0, actions, TH)
    assert out["decision"]["where"] == "DEAD_LEAK", out["decision"]


def test_motion_confusion_decision():
    ca0 = _toy_ca0(leak_dead=False, mix_motion=True)
    actions = np.zeros((12, 7, 2), dtype=np.float32)
    out = analyze_bank(ca0, actions, TH)
    assert out["decision"]["where"] == "MOTION_CONFUSION", out["occupancy"] | out["motion"] | out["decision"]


def test_true_dz_pose_control_does_not_silently_fail():
    ca0 = _toy_ca0(leak_dead=False, mix_motion=False)
    actions = np.zeros((12, 7, 2), dtype=np.float32)
    out = analyze_bank(ca0, actions, TH)
    r2 = out["r2"]["pose_from_true_dz"]
    assert r2 == r2 and r2 > 0.5, out["r2"]
    assert out["decision"]["physics"] != "CALIB_FAIL"


if __name__ == "__main__":
    test_dead_mask_matches_d1_rel_cut()
    print("ok test_dead_mask_matches_d1_rel_cut")
    test_energy_share_on_axis()
    print("ok test_energy_share_on_axis")
    test_dead_leak_decision()
    print("ok test_dead_leak_decision")
    test_motion_confusion_decision()
    print("ok test_motion_confusion_decision")
    test_true_dz_pose_control_does_not_silently_fail()
    print("ok test_true_dz_pose_control_does_not_silently_fail")
    print("all ok")
