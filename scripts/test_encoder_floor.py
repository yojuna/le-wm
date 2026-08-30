#!/usr/bin/env python3
"""Part A cuts and dump metrics (doc 16). No GPU."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from encoder_floor import (  # noqa: E402
    bracket_frac,
    decide_guard,
    dump_metrics,
    main as encoder_floor_main,
    onestep_errors,
)

TH = {
    "frac_accum_at_or_below": 0.25,
    "frac_infidelity_at_or_above": 0.5,
}


def test_decide_guard_frozen_cuts():
    assert decide_guard(0.1, calib_ok=True, th=TH) == "ACCUMULATION"
    assert decide_guard(0.25, calib_ok=True, th=TH) == "ACCUMULATION"
    assert decide_guard(0.4, calib_ok=True, th=TH) == "PARTIAL"
    assert decide_guard(0.5, calib_ok=True, th=TH) == "INFIDELITY"
    assert decide_guard(0.98, calib_ok=True, th=TH) == "INFIDELITY"
    assert decide_guard(0.1, calib_ok=False, th=TH) == "CALIB_FAIL"
    assert decide_guard(None, calib_ok=True, th=TH) == "CALIB_FAIL"


def test_bracket_frac_preview_shape():
    # Peeked A2: 1.21 / 1.23 ≈ 0.98. Frozen cuts are not retuned from this.
    f = bracket_frac(1.21, 0.0, 1.23)
    assert f is not None
    assert abs(f - 1.21 / 1.23) < 1e-9
    assert f >= 0.5  # would be INFIDELITY if GPU agrees
    assert bracket_frac(0.0, 0.0, 0.0) is None


def _toy_ca0(*, d_end_m1=1.43, onestep=0.4):
    rng = np.random.default_rng(0)
    n, l, d = 6, 10, 8
    z_true = rng.normal(size=(n, l, d)).astype(np.float32)
    z_star = z_true[:, -1] + 0.5
    noise = np.zeros_like(z_true)
    # Put a constant offset on predicted frames so onestep is known-ish.
    direction = np.zeros(d, dtype=np.float32)
    direction[0] = onestep
    noise[:, 3:] = direction
    hat1 = z_true + noise
    z_hat = np.stack([hat1, hat1], axis=1)
    d_end = np.zeros((n, 2), dtype=np.float32)
    d_end[:, 0] = d_end_m1
    d_end[:, 1] = 8.0
    return {
        "z_true": z_true,
        "z_star": z_star.astype(np.float32),
        "z_hat": z_hat.astype(np.float32),
        "m_values": np.array([1, 25], dtype=np.int32),
        "d_end": d_end,
        "d_start": np.full(n, 2.6, dtype=np.float32),
        "toward": np.zeros((n, 2), dtype=bool),
    }


def test_dump_metrics_keeps_two_columns():
    ca0 = _toy_ca0(d_end_m1=1.43, onestep=0.4)
    summary = {"by_m": {"1": {"mean_d_end": 1.43}}}
    m = dump_metrics(ca0, summary)
    assert abs(m["fork_mean_d_end_m1"] - 1.43) < 1e-5
    assert abs(m["fork_reported_d_end_m1"] - 1.43) < 1e-5
    # One-step is the planted 0.4, not d_end.
    assert abs(m["onestep_median"] - 0.4) < 1e-4
    assert abs(m["onestep_median"] - m["fork_mean_d_end_m1"]) > 0.5


def test_onestep_errors_history_slice():
    z_true = np.zeros((2, 6, 3), dtype=np.float64)
    z_hat = np.zeros_like(z_true)
    z_hat[:, 3:, 0] = 2.0
    err = onestep_errors(z_hat, z_true, history=3)
    assert err.shape == (2 * 3,)
    assert np.allclose(err, 2.0)


def test_dump_only_cli_writes_json():
    ca0 = _toy_ca0()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ca0_dir = td / "ca0"
        out_dir = td / "floor"
        ca0_dir.mkdir()
        np.savez_compressed(ca0_dir / "ca0.npz", **ca0)
        (ca0_dir / "summary.json").write_text(
            json.dumps({"by_m": {"1": {"mean_d_end": 1.43}}})
        )
        encoder_floor_main(
            [
                "--ca0",
                str(ca0_dir),
                "--out",
                str(out_dir),
                "--dump-only",
            ]
        )
        payload = json.loads((out_dir / "encoder_floor.json").read_text())
        copied = json.loads((ca0_dir / "encoder_floor.json").read_text())
        assert payload["xcheck_reproduces_d_end"] is True
        assert payload["guard_decision"] in {
            "ACCUMULATION",
            "PARTIAL",
            "INFIDELITY",
            "CALIB_FAIL",
        }
        assert "frac" in payload["bracket_metric"]
        assert payload["fork_metric"]["value"] == payload["xcheck_onestep_vs_d_end"]["fork_mean_d_end_m1"]
        assert copied["same_state_reencode_median"] is not None
        # Cuts in the written JSON must be the frozen file values, not retuned.
        th = payload["thresholds"]
        assert th["frac_accum_at_or_below"] == 0.25
        assert th["frac_infidelity_at_or_above"] == 0.5


if __name__ == "__main__":
    tests = [
        test_decide_guard_frozen_cuts,
        test_bracket_frac_preview_shape,
        test_dump_metrics_keeps_two_columns,
        test_onestep_errors_history_slice,
        test_dump_only_cli_writes_json,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("all ok")
