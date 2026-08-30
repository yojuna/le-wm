#!/usr/bin/env python3
"""Pre-retrain confirmation cuts (no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from infidelity_confirm import decide_confirm, frac_bundle, geometry_from_ca0  # noqa: E402

TH = {
    "frac_accum_at_or_below": 0.25,
    "frac_infidelity_at_or_above": 0.5,
    "pose_z_spearman_min": 0.30,
    "predicted_move_over_adjacent_identity_below": 0.25,
    "shuffle_gap_over_adjacent_deaf_below": 0.10,
}


def _ca0_identity(n=6, l=10, d=8):
    rng = np.random.default_rng(0)
    z = rng.normal(size=(n, l, d)).astype(np.float32)
    # Slow block motion so pose-z can correlate if we copy xy into state.
    st = np.zeros((n, l, 7), dtype=np.float32)
    t = np.arange(l, dtype=np.float32)
    st[:, :, 2] = 256.0 + t[None, :] * 0.5  # block x, away from walls
    st[:, :, 3] = 256.0
    st[:, :, 0] = 100.0  # agent far from block → free
    return {
        "z_true": z,
        "z_star": z[:, -1].copy(),
        "z_hat": np.stack([z, z], axis=1),  # perfect P
        "m_values": np.array([1, 25], dtype=np.int32),
        "path_state": st,
    }


def test_identity_hat_is_accumulation():
    b = frac_bundle(_ca0_identity()["z_true"], _ca0_identity()["z_true"])
    assert b["frac"] is not None
    assert b["frac"] <= 0.25
    assert b["onestep_median"] < 1e-6


def test_geometry_terciles_and_free_stratum():
    g = geometry_from_ca0(_ca0_identity())
    assert g["guard"] == "ACCUMULATION"
    assert g["predicted_move_over_adjacent"] > 0.5  # hat copies true motion
    assert g["strata"]["free"]["n"] == g["n_steps"]
    assert g["terciles"]["large"]["n"] > 0


def test_decide_seed_fluke_blocks_retrain():
    d = decide_confirm(
        {
            "geometry": {"frac": 0.98, "pose": {"spearman_adj_vs_pose": 0.8}, "predicted_move_over_adjacent": 0.9},
            "diverse": {"frac": 0.9},
            "seeds": [{"frac": 0.1}, {"frac": 0.9}],
            "shuffle": {"gap_over_adjacent": 0.5},
        },
        TH,
    )
    assert d["overall"] == "SEED_FLUKE"
    assert d["gate_part_b"] is False


def test_decide_bank_specific():
    d = decide_confirm(
        {
            "geometry": {"frac": 0.98, "pose": {"spearman_adj_vs_pose": 0.8}, "predicted_move_over_adjacent": 0.9},
            "diverse": {"frac": 0.1},
            "seeds": [{"frac": 0.9}, {"frac": 0.9}],
            "shuffle": {"gap_over_adjacent": 0.5},
        },
        TH,
    )
    assert d["overall"] == "BANK_SPECIFIC"
    assert d["gate_part_b"] is True


def test_decide_confirmed():
    d = decide_confirm(
        {
            "geometry": {"frac": 0.98, "pose": {"spearman_adj_vs_pose": 0.8}, "predicted_move_over_adjacent": 0.9},
            "diverse": {"frac": 0.9},
            "seeds": [{"frac": 0.9}, {"frac": 0.7}],
            "shuffle": {"gap_over_adjacent": 0.5},
        },
        TH,
    )
    assert d["overall"] == "CONFIRMED_INFIDELITY"
    assert d["gate_part_b"] is True


def test_decide_encoder_jitter():
    d = decide_confirm(
        {
            "geometry": {"frac": 0.98, "pose": {"spearman_adj_vs_pose": 0.05}, "predicted_move_over_adjacent": 0.9},
            "diverse": {"frac": 0.9},
            "seeds": [{"frac": 0.9}],
            "shuffle": {"gap_over_adjacent": 0.5},
        },
        TH,
    )
    assert d["overall"] == "ENCODER_JITTER"
    assert d["gate_part_b"] is False


def test_frozen_tercile_edges_not_refit():
    g = geometry_from_ca0(_ca0_identity(), tercile_edges=(1e9, 2e9))
    assert g["terciles"]["small"]["n"] == g["n_steps"]
    assert g["terciles"]["large"]["n"] == 0
    assert g["tercile_edges"][0] == 1e9


def test_decide_block_eval_three_ways():
    from block_motion_eval import decide_block_eval

    th = {
        "encoder_floor": {"frac_accum_at_or_below": 0.25, "frac_infidelity_at_or_above": 0.5},
        "b_eval_block": {"median_step_block_xy_min": 2.0},
        "b_eval_tercile": {"adj_q33": 0.83, "adj_q67": 1.74},
    }
    assert decide_block_eval(0.98, 2.1, th)["overall"] == "BLOCK_INFIDELITY"
    assert decide_block_eval(0.10, 0.12, th)["overall"] == "PUSHER_ONLY"
    assert decide_block_eval(0.40, 0.40, th)["overall"] == "BLOCK_PARTIAL"


if __name__ == "__main__":
    for fn in [
        test_identity_hat_is_accumulation,
        test_geometry_terciles_and_free_stratum,
        test_decide_seed_fluke_blocks_retrain,
        test_decide_bank_specific,
        test_decide_confirmed,
        test_decide_encoder_jitter,
        test_frozen_tercile_edges_not_refit,
        test_decide_block_eval_three_ways,
    ]:
        fn()
        print("ok", fn.__name__)
    print("all ok")
