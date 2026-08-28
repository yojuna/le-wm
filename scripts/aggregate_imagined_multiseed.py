#!/usr/bin/env python3
"""Aggregate imagined-φ multi-seed eval → JSON + docs summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CONDITIONS = ("E1_l2_c1", "E2_phi_v2", "E2_phi_imagined", "E4_random")
PROTOCOLS = ("short", "offset")
SEEDS = (0, 1, 2)


def load_metrics(root: Path, proto: str, seed: int, cond: str) -> dict:
    matches = list((root / proto / f"seed{seed}" / cond).rglob("metrics.json"))
    if not matches:
        raise FileNotFoundError(f"missing metrics for {proto}/seed{seed}/{cond}")
    return json.loads(matches[0].read_text())


def mean_min_pos(m: dict) -> float:
    return float(np.mean([e["min_pos_error"] for e in m["episodes"]]))


def summarize(root: Path) -> dict:
    out: dict = {"protocols": {}}
    for proto in PROTOCOLS:
        rows = {}
        for cond in CONDITIONS:
            succ, pos, state = [], [], []
            per_seed = []
            for seed in SEEDS:
                m = load_metrics(root, proto, seed, cond)
                agg = m["aggregate"]
                s = float(agg["success_rate"])
                p = mean_min_pos(m)
                st = float(agg["mean_min_state_distance"])
                succ.append(s)
                pos.append(p)
                state.append(st)
                per_seed.append(
                    {
                        "seed": seed,
                        "success_rate": s,
                        "num_successes": agg["num_successes"],
                        "num_episodes": agg["num_episodes"],
                        "mean_min_pos": p,
                        "mean_min_state": st,
                    }
                )
            rows[cond] = {
                "per_seed": per_seed,
                "success_mean": float(np.mean(succ)),
                "success_std": float(np.std(succ, ddof=1)) if len(succ) > 1 else 0.0,
                "mean_min_pos_mean": float(np.mean(pos)),
                "mean_min_state_mean": float(np.mean(state)),
            }
        e1 = rows["E1_l2_c1"]["success_mean"]
        e2v = rows["E2_phi_v2"]["success_mean"]
        e2i = rows["E2_phi_imagined"]["success_mean"]
        e4 = rows["E4_random"]["success_mean"]
        out["protocols"][proto] = {
            "conditions": rows,
            "checks": {
                "imagined_gt_v2": e2i > e2v,
                "imagined_ge_e1": e2i >= e1,
                "imagined_ge_e4": e2i >= e4,
                "delta_imagined_minus_v2_pp": e2i - e2v,
                "delta_imagined_minus_e1_pp": e2i - e1,
            },
        }
    off = out["protocols"]["offset"]["checks"]
    sh = out["protocols"]["short"]["checks"]
    out["go_nogo"] = {
        "offset_imagined_gt_v2": off["imagined_gt_v2"],
        "offset_imagined_ge_e1": off["imagined_ge_e1"],
        "short_not_collapsed": sh["imagined_ge_e1"],
    }
    return out


def markdown(agg: dict) -> str:
    lines = [
        "# Imagined-φ (H1) multi-seed results",
        "",
        "Weights: `lewm_phi_imagined_v1` vs `lewm_phi_v2` / L2 / random. Protocol: `06_imagined_phi.md`.",
        "",
        "## Go / no-go",
        "",
        "| Check | Result |",
        "|-------|--------|",
    ]
    g = agg["go_nogo"]
    lines += [
        f"| Offset imagined > v2 | {'PASS' if g['offset_imagined_gt_v2'] else 'FAIL'} |",
        f"| Offset imagined ≥ E1 | {'PASS' if g['offset_imagined_ge_e1'] else 'FAIL'} |",
        f"| Short imagined ≥ E1 | {'PASS' if g['short_not_collapsed'] else 'FAIL'} |",
        "",
    ]
    for proto in PROTOCOLS:
        lines += [
            f"## {proto}",
            "",
            "| Cond | success % (mean±std) | mean min pose | mean min state |",
            "|------|----------------------|---------------|----------------|",
        ]
        for cond in CONDITIONS:
            r = agg["protocols"][proto]["conditions"][cond]
            lines.append(
                f"| {cond} | {r['success_mean']:.1f}±{r['success_std']:.1f} | "
                f"{r['mean_min_pos_mean']:.2f} | {r['mean_min_state_mean']:.1f} |"
            )
        lines += ["", "### Per seed", "", "| Cond | seed0 | seed1 | seed2 |", "|------|-------|-------|-------|"]
        for cond in CONDITIONS:
            vals = [
                f"{s['success_rate']:.0f}% ({s['num_successes']}/{s['num_episodes']})"
                for s in agg["protocols"][proto]["conditions"][cond]["per_seed"]
            ]
            lines.append(f"| {cond} | " + " | ".join(vals) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    agg = summarize(args.root)
    (args.root / "aggregate.json").write_text(json.dumps(agg, indent=2))
    md = markdown(agg)
    docs = Path(__file__).resolve().parents[2] / "docs" / "lewm_phi_imagined_v1_summary.md"
    docs.write_text(md)
    print(md)
    print(f"wrote {docs}")


if __name__ == "__main__":
    main()
