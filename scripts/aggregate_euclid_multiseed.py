#!/usr/bin/env python3
"""Aggregate Euclidean multi-seed eval metrics → JSON + markdown summary stub."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


CONDITIONS = ("E1_l2_c1", "E2_phi_v2", "E4_random")
PROTOCOLS = ("short", "offset")
SEEDS = (0, 1, 2)


def load_metrics(root: Path, proto: str, seed: int, cond: str) -> dict:
    path = root / proto / f"seed{seed}" / cond / "pusht" / f"pusht_seed{seed}" / "metrics.json"
    if not path.exists():
        # eval_live sometimes uses pusht_seed0 naming — try glob
        matches = list((root / proto / f"seed{seed}" / cond).rglob("metrics.json"))
        if not matches:
            raise FileNotFoundError(path)
        path = matches[0]
    return json.loads(path.read_text())


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
        e2 = rows["E2_phi_v2"]["success_mean"]
        e4 = rows["E4_random"]["success_mean"]
        out["protocols"][proto] = {
            "conditions": rows,
            "checks": {
                "e2_gt_e1": e2 > e1,
                "e2_ge_e4": e2 >= e4,
                "e2_ge_e1": e2 >= e1,
                "delta_e2_minus_e1_pp": e2 - e1,
            },
        }
    short = out["protocols"]["short"]["checks"]
    offset = out["protocols"]["offset"]["checks"]
    out["go_nogo"] = {
        "short_replicate": short["e2_gt_e1"],
        "short_vs_random": short["e2_ge_e4"],
        "offset_transfer": offset["e2_ge_e1"],
    }
    return out


def markdown_table(agg: dict) -> str:
    lines = [
        "# Euclidean φ multi-seed results",
        "",
        f"**Aggregate:** generated from `lewm_phi_euclid_multiseed/aggregate.json`",
        "",
        "## Go / no-go",
        "",
        "| Check | Result |",
        "|-------|--------|",
    ]
    g = agg["go_nogo"]
    lines += [
        f"| Short E2 > E1 (replicate) | {'PASS' if g['short_replicate'] else 'FAIL'} |",
        f"| Short E2 ≥ E4 (vs random) | {'PASS' if g['short_vs_random'] else 'FAIL'} |",
        f"| Offset E2 ≥ E1 (transfer) | {'PASS' if g['offset_transfer'] else 'FAIL'} |",
        "",
    ]
    for proto in PROTOCOLS:
        lines += [f"## {proto}", "", "| Cond | success % (mean±std) | mean min pose | mean min state |", "|------|----------------------|---------------|----------------|"]
        for cond in CONDITIONS:
            r = agg["protocols"][proto]["conditions"][cond]
            lines.append(
                f"| {cond} | {r['success_mean']:.1f}±{r['success_std']:.1f} | "
                f"{r['mean_min_pos_mean']:.2f} | {r['mean_min_state_mean']:.1f} |"
            )
        lines.append("")
        lines.append("### Per seed")
        lines.append("")
        lines.append("| Cond | seed0 | seed1 | seed2 |")
        lines.append("|------|-------|-------|-------|")
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
    md = markdown_table(agg)
    # Write next to docs via relative path from le-wm
    docs = args.root.resolve().parents[3] / "docs" / "lewm_phi_euclid_multiseed_summary.md"
    # parents: euclid_multiseed -> pusht -> eval_results -> le-wm -> ws_le-wm
    # Actually: eval_results/pusht/lewm_phi_euclid_multiseed
    # parent[0]=pusht, [1]=eval_results, [2]=le-wm, [3]=ws_le-wm
    docs = Path(__file__).resolve().parents[2] / "docs" / "lewm_phi_euclid_multiseed_summary.md"
    docs.write_text(md)
    print(md)
    print(f"wrote {args.root / 'aggregate.json'}")
    print(f"wrote {docs}")


if __name__ == "__main__":
    main()
