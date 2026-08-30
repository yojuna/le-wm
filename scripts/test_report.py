#!/usr/bin/env python3
"""Guardrails for scripts/report.py (spec 15 v3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report import (  # noqa: E402
    compose_bluf,
    compose_section_verdicts,
    parse_tiers,
)
from viz import FigureResult, _sv, ca0_fork_from_by_m, load_thresholds  # noqa: E402


def _fr(tier, question, scalars, fork=None):
    return FigureResult(
        figure=object(),
        tier=tier,
        question=question,
        scalars=scalars,
        caption={
            "what": "x",
            "how_to_read": "y",
            "reading_here": "z",
            "would_overturn": "w",
        },
    )


def test_parse_tiers_default_one():
    assert parse_tiers("1") == {1}
    assert parse_tiers("1,2") == {1, 2}
    assert parse_tiers("all") == {1, 2, 3}


def test_bluf_matches_section_verdicts():
    th = load_thresholds()
    summary = {
        "by_m": {
            "1": {"frac_toward": 0.84, "mean_d_end": 1.43},
            "5": {"frac_toward": 0.62, "mean_d_end": 2.35},
        }
    }
    results = {
        "a1": _fr(
            1,
            "drift?",
            {
                "d_end_open": _sv(8.23),
                "d_start": _sv(2.61),
                "in_plane_drift_fraction": _sv(0.2, 0.35, "decorative"),
            },
        ),
        "a2": _fr(
            1,
            "fork?",
            {
                "median_onestep_error": _sv(1.1, 0.8, "fail"),
                "fork": _sv("CA0-INFIDELITY"),
            },
        ),
        "b1": _fr(
            1,
            "search?",
            {"signed_cost_gap": _sv(16.0, 0.0, "model")},
        ),
    }
    v = compose_section_verdicts(results, th, ca0_summary=summary)
    bluf = compose_bluf(v)
    assert v["qa"]["verdict"] == "DRIFT"
    assert v["qb"]["verdict"] == "CA0-INFIDELITY"
    assert v["qc"]["verdict"] == "MODEL"
    assert "INFIDELITY" in bluf
    assert "model scores the oracle" in bluf
    assert "{" not in bluf and "[" not in bluf
    for sec in ("qa", "qb", "qc"):
        assert v[sec]["text"].rstrip(".") in bluf or v[sec]["verdict"] in bluf


def test_infidelity_not_claimed_when_m1_passes():
    th = load_thresholds()
    summary = {
        "by_m": {
            "1": {"frac_toward": 0.95, "mean_d_end": 0.4},
            "5": {"frac_toward": 0.70, "mean_d_end": 2.0},
        }
    }
    assert ca0_fork_from_by_m({int(k): v for k, v in summary["by_m"].items()}, th)[
        "fork"
    ] != "CA0-INFIDELITY"
    results = {
        "a1": _fr(1, "q", {"d_end_open": _sv(2.0), "d_start": _sv(2.5), "in_plane_drift_fraction": _sv(0.5, 0.35, "informative")}),
        "a2": _fr(
            1,
            "q",
            {
                "median_onestep_error": _sv(0.2, 0.8, "pass"),
                "fork": _sv("CA0-INFIDELITY"),  # stale / wrong claim
            },
        ),
        "b1": _fr(1, "q", {"signed_cost_gap": _sv(-1.0, 0.0, "search")}),
    }
    v = compose_section_verdicts(results, th, ca0_summary=summary)
    bluf = compose_bluf(v)
    assert v["qb"]["verdict"] != "CA0-INFIDELITY"
    assert "INFIDELITY" not in bluf


def test_report_source_does_not_stitch_captions_json():
    src = (ROOT / "scripts" / "report.py").read_text()
    assert "captions.json" not in src
    assert "fig_a1" in src and "fig_a2" in src and "fig_b1" in src


def test_b1_not_a_height_map():
    src = (ROOT / "viz.py").read_text().lower()
    assert "height map" not in src
    assert "pcolormesh" not in src


def test_qc_wrong_objective_not_underbudget():
    th = load_thresholds()
    results = {
        "a1": _fr(1, "q", {"d_end_open": _sv(8.23), "d_start": _sv(2.61)}),
        "a2": _fr(1, "q", {"median_onestep_error": _sv(1.2, 0.8, "fail"), "fork": _sv("CA0-INFIDELITY")}),
        "b1": _fr(1, "q", {"signed_cost_gap": _sv(16.0, 0.0, "model")}),
        "b2": _fr(2, "q", {"still_improving_at_end": _sv(True, None, "wrong-objective")}),
    }
    summary = {"by_m": {"1": {"frac_toward": 0.84, "mean_d_end": 1.43}}}
    v = compose_section_verdicts(results, th, ca0_summary=summary)
    assert "further from the oracle" in v["qc"]["text"]
    assert "under-budget" not in v["qc"]["text"].lower()


if __name__ == "__main__":
    test_parse_tiers_default_one()
    test_bluf_matches_section_verdicts()
    test_infidelity_not_claimed_when_m1_passes()
    test_report_source_does_not_stitch_captions_json()
    test_b1_not_a_height_map()
    test_qc_wrong_objective_not_underbudget()
    print("all report tests passed")
