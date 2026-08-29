#!/usr/bin/env python3
"""CPU tests for Phase B dump/probe/drift helpers and --horizon plumbing."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase_b import (  # noqa: E402
    CEM_HORIZON,
    HISTORY,
    load_dump,
    pad_action,
    remaining_pose_error,
    save_dump,
    shuffle_future_actions,
    summarize_drift,
)
from eval_live import EnvSpec, _apply_horizon  # noqa: E402


def test_pad_action():
    a = np.array([1.0, 2.0], dtype=np.float32)
    p = pad_action(a, action_dim=10)
    assert p.shape == (10,)
    assert p[0] == 1.0 and p[1] == 2.0 and p[9] == 0.0


def test_shuffle_keeps_history():
    rng = np.random.default_rng(0)
    acts = np.arange(20, dtype=np.float32).reshape(10, 2)
    sh = shuffle_future_actions(acts, history=3, rng=rng)
    assert np.allclose(sh[:3], acts[:3])
    assert sh.shape == acts.shape
    # tail permutation of original tail (same multiset)
    assert sorted(sh[3:].reshape(-1).tolist()) == sorted(acts[3:].reshape(-1).tolist())


def test_summarize_drift_excludes_history():
    err = np.zeros(8, dtype=np.float64)
    err[HISTORY:] = np.arange(1, 8 - HISTORY + 1, dtype=np.float64)
    s = summarize_drift(err, history=HISTORY)
    assert s["mean_all_frames"] < s["mean_predicted_only"]
    cem_idx = HISTORY + CEM_HORIZON - 1
    assert s["at_h5_index"] == float(err[cem_idx])


def test_remaining_pose_zero_at_end():
    states = np.zeros((5, 7), dtype=np.float64)
    states[:, 0] = np.linspace(0, 10, 5)
    rem = remaining_pose_error("pusht", states)
    assert rem[-1] == 0.0
    assert rem[0] > rem[-2]


def test_linear_probe_recovers_linear_factor():
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    z = rng.normal(size=(400, 32))
    w = rng.normal(size=32)
    y = z @ w
    tr, va = z[:300], z[300:]
    ytr, yva = y[:300], y[300:]
    model = make_pipeline(StandardScaler(), Ridge(alpha=0.1))
    model.fit(tr, ytr)
    r2 = r2_score(yva, model.predict(va))
    assert r2 > 0.95, r2


def test_dump_roundtrip(tmp_path: Path | None = None):
    out = Path("/tmp/phase_b_dump_test") if tmp_path is None else tmp_path
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dump.npz"
    z = np.random.randn(4, 6, 8).astype(np.float32)
    save_dump(
        path,
        {
            "z": z,
            "remaining_k": np.arange(4 * 6).reshape(4, 6).astype(np.float32),
            "meta": {"env": "pusht", "n_segments": 4},
        },
    )
    loaded = load_dump(path)
    assert loaded["z"].shape == (4, 6, 8)
    assert loaded["meta"]["env"] == "pusht"


def test_apply_horizon():
    spec = EnvSpec(
        env_name="swm/PushT-v1",
        hf_repo="x",
        ckpt_dir="hf_pusht",
        horizon=5,
        receding_horizon=5,
    )

    @dataclass
    class A:
        horizon: int = 0
        receding_horizon: int = 0

    assert _apply_horizon(spec, A()).horizon == 5
    s2 = _apply_horizon(spec, A(horizon=2))
    assert s2.horizon == 2 and s2.receding_horizon == 2
    s3 = _apply_horizon(spec, A(horizon=8, receding_horizon=3))
    assert s3.horizon == 8 and s3.receding_horizon == 3


def test_episode_len_pixels_fallback():
    from eval_logging.pairs import EpisodeTraj

    ep = EpisodeTraj(seed=0, pixels=[np.zeros((4, 4, 3))] * 3)
    assert len(ep) == 3


if __name__ == "__main__":
    test_pad_action()
    test_shuffle_keeps_history()
    test_summarize_drift_excludes_history()
    test_remaining_pose_zero_at_end()
    test_linear_probe_recovers_linear_factor()
    test_dump_roundtrip()
    test_apply_horizon()
    test_episode_len_pixels_fallback()
    print("all phase_b tests passed")
