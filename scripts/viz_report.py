#!/usr/bin/env python3
"""Static HTML/markdown campaign report stitching metrics + viz captions.

  python scripts/viz_report.py \\
      --out eval_results/pusht/viz_report \\
      --ca0 eval_results/pusht/ca0_closed_loop/seed0 \\
      --ca1 eval_results/pusht/ca1_drift_contacts/seed0/drift_by_event.json \\
      --figs eval_results/pusht/viz/seed0 \\
      --probe eval_results/pusht/phase_b_dump/seed0/probe_summary.json \\
      --drift eval_results/pusht/phase_b_dump/seed0/drift_summary.json \\
      --oracle-imagine eval_results/pusht/c0_oracle_livebank/seed0/oracle_imagine.json \\
      --cem-metrics eval_results/pusht/c0_livebank_cem/metrics.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(path: Path | None):
    if path is None or not Path(path).exists():
        return None
    p = Path(path)
    if p.is_dir():
        cand = p / "summary.json"
        if cand.exists():
            p = cand
        else:
            return None
    return json.loads(p.read_text())


def _esc(x) -> str:
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _collect_captions(fig_dirs: list[Path]) -> dict:
    out: dict = {}
    for d in fig_dirs:
        cap_path = d / "captions.json" if d.is_dir() else d
        if not cap_path.exists():
            continue
        blob = json.loads(cap_path.read_text())
        if isinstance(blob, dict):
            out.update(blob)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ca0", type=Path, default=None)
    p.add_argument("--ca1", type=Path, default=None)
    p.add_argument("--figs", type=Path, nargs="*", default=[])
    p.add_argument("--probe", type=Path, default=None)
    p.add_argument("--drift", type=Path, default=None)
    p.add_argument("--oracle-imagine", type=Path, default=None)
    p.add_argument("--cem-metrics", type=Path, default=None)
    p.add_argument("--sweep", type=Path, default=None)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_out = out / "figs"
    fig_out.mkdir(parents=True, exist_ok=True)

    ca0 = _load(args.ca0)
    ca1 = _load(args.ca1)
    captions = _collect_captions([Path(x) for x in args.figs])
    probe = _load(args.probe)
    drift = _load(args.drift)
    imagine = _load(args.oracle_imagine)
    cem = _load(args.cem_metrics)
    sweep = _load(args.sweep)

    md = ["# lewm-phi visual report", "", "Scalars are the citable result; figures navigate.", ""]

    if ca0:
        md += [
            "## CA0 fork",
            "",
            f"**{ca0.get('fork')}** — {ca0.get('fork_reason')}",
            "",
            "```json",
            json.dumps(ca0.get("by_m"), indent=2),
            "```",
            "",
        ]
    if imagine:
        md += [
            "## C0 oracle-imagine (open-loop)",
            "",
            f"toward-goal **{imagine.get('frac_imag_moved_toward_goal')}** "
            f"mean ‖ẑ_end−z*‖ **{imagine.get('mean_d_hat_end')}** "
            f"(n={imagine.get('n')})",
            "",
        ]
    if cem and "aggregate" in cem:
        ag = cem["aggregate"]
        md += [
            "## CEM on live-bank",
            "",
            f"success_rate **{ag.get('success_rate')}**  n={ag.get('num_episodes')}",
            "",
        ]
    if probe:
        md += [
            "## B1 probes",
            "",
            f"mean linear R² **{probe.get('mean_state_linear_r2')}**  "
            f"D6 **{probe.get('d6_recommendation')}**",
            "",
        ]
    if drift:
        t = drift.get("true") or {}
        md += [
            "## B2 drift",
            "",
            f"predicted-only **{t.get('mean_predicted_only')}**  at h=5 **{t.get('at_h5_index')}**",
            "",
        ]
    if ca1:
        md += ["## CA1 contact vs free", "", "```json", json.dumps(ca1, indent=2)[:4000], "```", ""]
    if sweep:
        md += ["## CA3 sweep", "", f"factor `{sweep.get('factor')}`", ""]

    md += ["## Figures", ""]
    html_figs = []
    for name, cap in captions.items():
        png = cap.get("png")
        pdf = cap.get("pdf")
        motivates = cap.get("motivates")
        scalars = cap.get("scalars")
        md += [
            f"### {name}",
            "",
            f"Motivates: {motivates}",
            "",
            "```json",
            json.dumps(scalars, indent=2, default=str),
            "```",
            "",
        ]
        rel = None
        if png and Path(png).exists():
            dest = fig_out / Path(png).name
            shutil.copy2(png, dest)
            rel = f"figs/{dest.name}"
            md.append(f"![{name}]({rel})")
            md.append("")
            if pdf and Path(pdf).exists():
                shutil.copy2(pdf, fig_out / Path(pdf).name)
        html_figs.append(
            f"<h3>{_esc(name)}</h3><p>Motivates: {_esc(motivates)}</p>"
            f"<pre>{_esc(json.dumps(scalars, indent=2, default=str))}</pre>"
            + (f"<img src='{_esc(rel)}' alt='{_esc(name)}' style='max-width:100%'/>" if rel else "")
        )

    (out / "report.md").write_text("\n".join(md))
    body = "\n".join(html_figs) if html_figs else "<p>No figures yet. Run scripts/viz.py.</p>"
    fork = None if not ca0 else ca0.get("fork")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>lewm-phi visual report</title>
<style>body{{font-family:sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}}
pre{{background:#111;color:#eee;padding:0.8rem;overflow:auto}} img{{border:1px solid #ccc}}</style>
</head><body>
<h1>lewm-phi visual report</h1>
<p>Scalars are the citable result; figures navigate. See report.md for the metric pull.</p>
<pre>{_esc(json.dumps({"ca0_fork": fork, "ca0": None if not ca0 else ca0.get("by_m")}, indent=2))}</pre>
{body}
</body></html>
"""
    (out / "index.html").write_text(html)
    print(f"wrote {out / 'index.html'} and {out / 'report.md'}")


if __name__ == "__main__":
    main()
