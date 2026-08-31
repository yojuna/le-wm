#!/usr/bin/env python3
"""Scientific diagnostic report from dumps (spec 15 v3).

Calls fig_* on dumps — never stitches a pre-baked caption gallery.

  python scripts/report.py --ca0 eval_results/pusht/ca0_closed_loop/seed0 \\
      --cem-capture eval_results/pusht/cem_capture/seed0 \\
      --dump eval_results/pusht/phase_b_dump/seed0 \\
      --tiers 1 --out eval_results/pusht/viz_report/seed0
"""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as _dt
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase_b import DUMP_VERSION  # noqa: E402
from viz import (  # noqa: E402
    FigureResult,
    ca0_fork_from_by_m,
    fig_a1,
    fig_a2,
    fig_a3,
    fig_a4,
    fig_a5,
    fig_b1,
    fig_b2,
    fig_b3,
    fig_c2,
    fig_d1,
    fig_d2,
    fig_e1,
    fig_f1,
    load_ca0,
    load_cem_capture,
    load_dump,
    load_thresholds,
    save_figure,
)

SECTIONS = [
    ("qa", "Q-A  Does the rollout drift?", ["a1"], 1),
    ("qb", "Q-B  Accumulation or per-step infidelity?", ["a2"], 1),
    ("qc", "Q-C  Search or model problem?", ["b1", "b2"], 1),
    ("qd", "Q-D  Why / where does it drift?", ["a3", "a4", "a5"], 2),
    ("qe", "Q-E  Objective and representation?", ["d1", "c2", "d2"], 2),
    ("qf", "Q-F  Task geometry?", ["e1"], 3),
]
APPENDIX_FIGS = ["f1", "b3"]
FIG_TIER = {
    "a1": 1,
    "a2": 1,
    "b1": 1,
    "b2": 2,
    "a3": 2,
    "d1": 2,
    "a4": 3,
    "a5": 2,
    "b3": 3,
    "c2": 3,
    "d2": 3,
    "e1": 3,
    "f1": 3,
}


def parse_tiers(spec: str) -> set[int]:
    s = spec.strip().lower()
    if s in ("all", "1,2,3"):
        return {1, 2, 3}
    out = set()
    for part in s.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out or {1}


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _esc(x) -> str:
    return html.escape("" if x is None else str(x))


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if v != v:
            return "—"
        if abs(v) >= 100:
            return f"{v:.1f}"
        return f"{v:.3g}"
    return str(v)


def _scalar_value(sv):
    if isinstance(sv, dict):
        return sv.get("value")
    return sv


def compose_section_verdicts(
    results: dict[str, FigureResult],
    thresholds: dict,
    *,
    ca0_summary: dict | None = None,
) -> dict[str, dict[str, str]]:
    """One verdict per question. BLUF is the composition of these."""
    th = thresholds or {}
    ca0th = th.get("ca0", {})
    out: dict[str, dict[str, str]] = {}

    if "a1" in results:
        a1 = results["a1"].scalars
        d_end = _scalar_value(a1.get("d_end_open"))
        d_start = _scalar_value(a1.get("d_start"))
        plane = a1.get("in_plane_drift_fraction") or {}
        drifts = (
            isinstance(d_end, (int, float))
            and isinstance(d_start, (int, float))
            and d_end > d_start
        )
        deco = isinstance(plane, dict) and plane.get("verdict") == "decorative"
        if drifts:
            text = (
                f"Bank-mean open-loop end-dist {_fmt(d_end)} exceeds start {_fmt(d_start)}: "
                "the imagined rollout drifts."
            )
        else:
            text = (
                f"Open-loop end-dist {_fmt(d_end)} vs start {_fmt(d_start)}: "
                "drift is not established on the bank mean."
            )
        if deco:
            text += " The PCA overlay is decorative; trust the full-space curve."
        out["qa"] = {"verdict": "DRIFT" if drifts else "NO-DRIFT", "text": text}

    if "a2" in results:
        by = {}
        if ca0_summary and ca0_summary.get("by_m"):
            by = {int(k): v for k, v in ca0_summary["by_m"].items()}
        if by:
            fork = ca0_fork_from_by_m(by, th)
        else:
            claimed = _scalar_value(results["a2"].scalars.get("fork"))
            fork = {"fork": claimed or "CA0-AMBIGUOUS", "reason": ""}
        m1 = by.get(1)
        if m1 is not None:
            toward = float(m1.get("frac_toward", m1.get("toward", 0)))
            dend = float(m1["mean_d_end"])
            if toward >= float(ca0th.get("m1_toward_guard", 0.90)) and dend <= float(
                ca0th.get("m1_d_end_guard", 1.0)
            ):
                if fork["fork"] == "CA0-INFIDELITY":
                    fork = {
                        "fork": "CA0-AMBIGUOUS",
                        "reason": "m=1 teacher-force guard passes; INFIDELITY is not claimed.",
                    }
        med = results["a2"].scalars.get("median_onestep_error") or {}
        med_v = _scalar_value(med)
        text = (
            f"Median one-step error {_fmt(med_v)} "
            f"(cut {_fmt(med.get('threshold') if isinstance(med, dict) else None)}). "
            f"Fork {fork['fork']}"
            + (f": {fork['reason']}" if fork.get("reason") else ".")
        )
        out["qb"] = {"verdict": str(fork["fork"]), "text": text}

    if "b1" in results:
        gap = results["b1"].scalars.get("signed_cost_gap") or {}
        val = _scalar_value(gap)
        v = gap.get("verdict") if isinstance(gap, dict) else None
        if v == "model":
            text = (
                f"Signed cost gap {_fmt(val)} is positive: the model scores the oracle "
                "worse than the CEM pick (model problem, not a search miss)."
            )
            label = "MODEL"
            if "b2" in results:
                sv = results["b2"].scalars.get("still_improving_at_end") or {}
                b2v = sv.get("verdict") if isinstance(sv, dict) else None
                if b2v == "wrong-objective":
                    text += (
                        " CEM is still lowering cost; more iterations would move further "
                        "from the oracle, not toward it."
                    )
                elif b2v == "converged-wrong-objective":
                    text += " CEM already converged on that mis-ranked score; adding iterations is not the fix."
        elif v == "search":
            text = (
                f"Signed cost gap {_fmt(val)} is not positive: CEM failed to find a "
                "better-scoring action (search problem)."
            )
            label = "SEARCH"
        else:
            text = "Oracle cost missing; search vs model is not scored."
            label = "UNKNOWN"
        out["qc"] = {"verdict": label, "text": text}

    if "a3" in results:
        ratio = results["a3"].scalars.get("ratio_contact_over_free") or {}
        rv = _scalar_value(ratio)
        spike = isinstance(ratio, dict) and ratio.get("verdict") == "spike"
        text = (
            f"Contact/free drift ratio {_fmt(rv)}"
            + (" exceeds the spike cut — contact-driven." if spike else " — not a contact spike.")
        )
        pose = _scalar_value(results["a3"].scalars.get("pose_probe_share_sum"))
        if pose is not None:
            text += f" Linear pose-probe directions explain {_fmt(pose)} of drift energy (not a block-vs-agent split)."
        bank = _scalar_value(results["a3"].scalars.get("bank"))
        if bank:
            text += f" A3 bank: {bank}."
        if "a5" in results:
            ang = _scalar_value(results["a5"].scalars.get("mean_angular_error_deg"))
            text += f" Bank-mean full-space action-effect angle {_fmt(ang)}°."
        out["qd"] = {"verdict": "CONTACT-SPIKE" if spike else "STEADY", "text": text}

    if "d1" in results:
        dead = results["d1"].scalars.get("dead_dim_fraction") or {}
        nid = _scalar_value(results["d1"].scalars.get("nonlinear_id"))
        pr = _scalar_value(results["d1"].scalars.get("participation_ratio"))
        hard = isinstance(dead, dict) and dead.get("verdict") == "hard-elbow"
        text = (
            f"Participation ratio {_fmt(pr)}, nonlinear ID {_fmt(nid)}, "
            f"dead-dim {_fmt(_scalar_value(dead))}"
            + (" — hard elbow, do not scale." if hard else " — not a hard dead-tail.")
        )
        out["qe"] = {"verdict": "HARD-ELBOW" if hard else "SOFT", "text": text}

    if "e1" in results:
        rate = _scalar_value(results["e1"].scalars.get("cem_success_rate"))
        out["qf"] = {
            "verdict": "MAPPED",
            "text": f"CEM success rate {_fmt(rate)} over live-bank goal poses.",
        }
    return out


def compose_bluf(verdicts: dict[str, dict[str, str]]) -> str:
    """2–4 plain-language sentences. No JSON."""
    parts = []
    qa = verdicts.get("qa")
    qb = verdicts.get("qb")
    qc = verdicts.get("qc")
    if qa:
        parts.append(qa["text"].rstrip("."))
    if qb:
        parts.append(qb["text"].rstrip("."))
    if qc:
        parts.append(qc["text"].rstrip("."))
    fork = (qb or {}).get("verdict", "")
    planner = (qc or {}).get("verdict", "")
    if fork == "CA0-INFIDELITY" and planner == "MODEL":
        parts.append("Next: CA-train is motivated; C1 stays gated. Do not retune the CA0 cuts")
    elif fork == "CA0-ACCUMULATION":
        parts.append("Next: treat this as compounding; protocol/re-encode, and C1 can be re-opened")
    elif fork == "CA0-AMBIGUOUS":
        parts.append("Next: the fork is not claimed; do not start CA-train or C1 from this report")
    sentences = [p.strip() + "." for p in parts if p]
    return " ".join(sentences[:4])


def _caption_block(result: FigureResult) -> tuple[str, str]:
    cap = result.caption or {}
    md = (
        f"**What this is.** {cap.get('what', '')}\n\n"
        f"**How to read it.** {cap.get('how_to_read', '')}\n\n"
        f"**Reading here.** {cap.get('reading_here', '')}\n\n"
        f"**What would overturn this.** {cap.get('would_overturn', '')}\n"
    )
    html_b = (
        "<dl class='cap'>"
        f"<dt>What this is</dt><dd>{_esc(cap.get('what', ''))}</dd>"
        f"<dt>How to read it</dt><dd>{_esc(cap.get('how_to_read', ''))}</dd>"
        f"<dt>Reading here</dt><dd>{_esc(cap.get('reading_here', ''))}</dd>"
        f"<dt>What would overturn this</dt><dd>{_esc(cap.get('would_overturn', ''))}</dd>"
        "</dl>"
    )
    return md, html_b


def _load_e1(oracle_bank: Path, cem_episodes: Path):
    from eval_logging.oracle_bank import load_oracle_bank
    import numpy as np

    pairs, _ = load_oracle_bank(oracle_bank)
    rows = list(csv.DictReader(Path(cem_episodes).open()))
    n = min(len(pairs), len(rows))
    xy, suc = [], []
    for i in range(n):
        st = np.asarray(
            pairs[i].path_state if pairs[i].path_state is not None else pairs[i].goal_state
        )
        xy.append(st[2:4] if st.ndim == 1 else st[-1, 2:4])
        suc.append(str(rows[i].get("success", "")).lower() in ("true", "1", "yes"))
    return np.stack(xy), np.asarray(suc)


def collect_results(args, tiers: set[int]) -> tuple[dict[str, FigureResult], dict]:
    """Call fig_* on dumps. Skip figures whose dump is missing."""
    results: dict[str, FigureResult] = {}
    ca0, ca0_sum = (None, {})
    if args.ca0:
        ca0, ca0_sum = load_ca0(args.ca0)
        floor_path = Path(args.ca0) / "encoder_floor.json"
        if floor_path.exists():
            ca0_sum = dict(ca0_sum)
            ca0_sum["encoder_floor"] = json.loads(floor_path.read_text())
    dump = load_dump(args.dump) if args.dump else None
    capture = load_cem_capture(args.cem_capture) if args.cem_capture else None
    wanted = {k for k, t in FIG_TIER.items() if t in tiers}

    def need(name: str) -> bool:
        return name in wanted

    if need("a1") and ca0 is not None:
        results["a1"] = fig_a1(ca0, pair=args.pair, summary=ca0_sum)
    if need("a2") and ca0 is not None:
        results["a2"] = fig_a2(ca0, summary=ca0_sum)
    if need("b1") and capture is not None:
        results["b1"] = fig_b1(capture)
    if need("b2") and capture is not None:
        results["b2"] = fig_b2(capture)
    if need("b3") and capture is not None:
        results["b3"] = fig_b3(capture)
    if need("a3"):
        src = ca0 if ca0 is not None else dump
        if src is not None:
            results["a3"] = fig_a3(src, segment=args.segment)
    if need("a4") and ca0 is not None and args.oracle_bank:
        from eval_logging.oracle_bank import load_oracle_bank
        import numpy as np

        pairs, _ = load_oracle_bank(args.oracle_bank)
        acts = []
        for p in pairs[: len(ca0["z_true"])]:
            a = np.asarray(p.oracle_actions, dtype=np.float32)
            if a.ndim == 1:
                a = a.reshape(1, -1)
            acts.append(a)
        maxlen = max(len(a) for a in acts)
        adim = acts[0].shape[-1]
        stacked = np.zeros((len(acts), maxlen, adim), dtype=np.float32)
        for i, a in enumerate(acts):
            stacked[i, : len(a)] = a
        results["a4"] = fig_a4(ca0, stacked)
    if need("a5") and ca0 is not None:
        results["a5"] = fig_a5(ca0, pair=args.pair)
    if need("d1") and dump is not None:
        results["d1"] = fig_d1(dump)
    if need("c2") and dump is not None:
        results["c2"] = fig_c2(dump)
    if need("d2") and args.sweep:
        results["d2"] = fig_d2(json.loads(Path(args.sweep).read_text()))
    if need("f1") and ca0 is not None and args.oracle_bank:
        from eval_logging.oracle_bank import load_oracle_bank

        pairs, _ = load_oracle_bank(args.oracle_bank)
        pair = pairs[args.pair]
        m_values = [int(x) for x in ca0["m_values"]]
        mi = m_values.index(25) if 25 in m_values else len(m_values) - 1
        results["f1"] = fig_f1(pair.path_pixels, ca0["z_true"][args.pair], ca0["z_hat"][args.pair, mi])
    if need("e1") and args.cem_episodes and args.oracle_bank:
        xy, suc = _load_e1(args.oracle_bank, args.cem_episodes)
        results["e1"] = fig_e1(goal_xy=xy, success=suc)
    return results, ca0_sum


def _provenance(args, ca0_sum: dict) -> dict[str, Any]:
    meta = {}
    if args.dump:
        mp = Path(args.dump)
        meta_path = mp / "dump.meta.json" if mp.is_dir() else mp.with_suffix(".meta.json")
        if not meta_path.exists() and mp.is_dir():
            meta_path = mp / "dump.meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
    bank = (ca0_sum or {}).get("bank_meta") or {}
    return {
        "date": _dt.date.today().isoformat(),
        "git_sha": _git_sha(),
        "dump_version": meta.get("version", DUMP_VERSION),
        "seed": bank.get("seed", meta.get("seed")),
        "ckpt": meta.get("ckpt") or "hf_pusht",
        "ca0": str(args.ca0) if args.ca0 else None,
        "dump": str(args.dump) if args.dump else None,
        "cem_capture": str(args.cem_capture) if args.cem_capture else None,
        "oracle_bank": str(args.oracle_bank) if args.oracle_bank else None,
        "n_pairs": (ca0_sum or {}).get("n_pairs"),
        "fork_json": (ca0_sum or {}).get("fork"),
    }


def _scalars_rows(results: dict[str, FigureResult]) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for fid in ["a1", "a2", "b1", "b2", "a3", "d1", "a4", "a5", "b3", "c2", "d2", "e1", "f1"]:
        if fid not in results:
            continue
        for name, sv in results[fid].scalars.items():
            if name in ("pair", "segment", "fork_reason", "n", "example_pair"):
                continue
            if not isinstance(sv, dict):
                rows.append((fid, name, _fmt(sv), "—", "—"))
                continue
            rows.append(
                (fid, name, _fmt(sv.get("value")), _fmt(sv.get("threshold")), _fmt(sv.get("verdict")))
            )
    return rows


def render_report(
    results: dict[str, FigureResult],
    verdicts: dict[str, dict[str, str]],
    provenance: dict,
    *,
    tiers: set[int],
    fig_dir: Path,
    bluf: str,
) -> tuple[str, str]:
    """Return (markdown, html)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    png_rel: dict[str, str] = {}
    png_b64: dict[str, str] = {}
    for fid, res in results.items():
        stem = fig_dir / fid
        save_figure(res, stem)
        png_rel[fid] = f"figures/{fid}.png"
        png_b64[fid] = base64.b64encode((stem.with_suffix(".png")).read_bytes()).decode("ascii")

    def fig_block(fid: str) -> tuple[str, str]:
        res = results[fid]
        md_cap, html_cap = _caption_block(res)
        md = (
            f"#### {fid.upper()} — {res.question}\n\n"
            f"{md_cap}\n"
            f"![{fid}]({png_rel[fid]})\n"
        )
        img = (
            f"<img src='data:image/png;base64,{png_b64[fid]}' alt='{_esc(fid)}'/>"
            if fid in png_b64
            else ""
        )
        ht = (
            f"<article class='fig' id='fig-{_esc(fid)}'>"
            f"<h4>{_esc(fid.upper())} — {_esc(res.question)}</h4>"
            f"{html_cap}{img}</article>"
        )
        return md, ht

    rows = _scalars_rows(results)
    md_table = [
        "| Fig | Scalar | Value | Threshold | Verdict |",
        "|-----|--------|------:|-----------|---------|",
    ]
    html_rows = []
    for fid, name, val, cut, verd in rows:
        md_table.append(f"| {fid} | `{name}` | {val} | {cut} | {verd} |")
        cls = "fail" if verd in ("fail", "model", "decorative", "spike", "hard-elbow") else ""
        html_rows.append(
            f"<tr class='{cls}'><td>{_esc(fid)}</td><td><code>{_esc(name)}</code></td>"
            f"<td>{_esc(val)}</td><td>{_esc(cut)}</td><td>{_esc(verd)}</td></tr>"
        )

    prov_bits = [
        f"{k}={v}" for k, v in provenance.items() if v not in (None, "")
    ]
    banner = " · ".join(prov_bits)
    md = [
        "# Diagnostic report (lewm-phi)",
        "",
        f"*{banner}*",
        "",
        "## Bottom line",
        "",
        bluf,
        "",
        "## Scalars",
        "",
        *md_table,
        "",
    ]
    html_sections = []
    t3_md = []
    t3_html = []

    for sid, title, fids, min_tier in SECTIONS:
        present = [f for f in fids if f in results]
        if not present:
            continue
        v = verdicts.get(sid)
        body_md = [f"## {title}", ""]
        if v:
            body_md += [f"**Verdict.** {v['text']}", ""]
        body_html = [f"<section id='{_esc(sid)}'><h2>{_esc(title)}</h2>"]
        if v:
            body_html.append(f"<p class='sverdict'><strong>Verdict.</strong> {_esc(v['text'])}</p>")
        for fid in present:
            m, h = fig_block(fid)
            if FIG_TIER[fid] >= 3:
                t3_md += [m]
                t3_html.append(h)
            else:
                body_md.append(m)
                body_html.append(h)
        body_html.append("</section>")
        if FIG_TIER[present[0]] >= 3:
            continue
        md.extend(body_md)
        html_sections.append("".join(body_html))

    for fid in APPENDIX_FIGS:
        if fid in results and FIG_TIER[fid] >= 3:
            m, h = fig_block(fid)
            t3_md.append(m)
            t3_html.append(h)

    if t3_md and 3 in tiers:
        md += ["## Appendix (Tier 3)", "", *t3_md]
        html_sections.append(
            "<details class='t3'><summary>Appendix (Tier 3)</summary>"
            + "".join(t3_html)
            + "</details>"
        )

    md_text = "\n".join(md) + "\n"
    html_text = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Diagnostic report — lewm-phi</title>
<style>
body{{font-family:Georgia, "Iowan Old Style", Palatino, serif; max-width:42rem;
  margin:2rem auto; padding:0 1.2rem 3rem; line-height:1.45; color:#1a1a1a;}}
h1,h2,h3,h4{{font-weight:600; line-height:1.25;}}
h1{{font-size:1.55rem;}} h2{{font-size:1.2rem; margin-top:2rem;}}
.banner{{font-size:0.82rem; color:#555;}}
.bluf{{background:#f4f1ea; border-left:4px solid #333; padding:0.95rem 1.1rem; margin:1.1rem 0;}}
.bluf p{{margin:0;}}
table{{border-collapse:collapse; width:100%; font-size:0.9rem; margin:0.6rem 0 1.2rem;}}
th,td{{border-bottom:1px solid #ddd; padding:0.32rem 0.45rem; text-align:left;}}
th{{font-size:0.78rem; text-transform:uppercase; letter-spacing:0.03em; color:#444;}}
tr.fail td:last-child{{font-weight:600;}}
.sverdict{{margin:0.4rem 0 0.8rem;}}
.fig img{{max-width:100%; height:auto; border:1px solid #ddd;}}
.cap dt{{font-weight:600; margin-top:0.45rem;}}
.cap dd{{margin:0.1rem 0 0 0;}}
details.t3{{margin-top:2rem;}}
summary{{cursor:pointer; font-weight:600;}}
code{{font-size:0.85em;}}
</style></head><body>
<h1>Diagnostic report</h1>
<p class="banner">{_esc(banner)}</p>
<h2>Bottom line</h2>
<div class="bluf"><p>{_esc(bluf)}</p></div>
<h2>Scalars</h2>
<table><thead><tr><th>Fig</th><th>Scalar</th><th>Value</th><th>Threshold</th><th>Verdict</th></tr></thead>
<tbody>
{''.join(html_rows)}
</tbody></table>
{''.join(html_sections)}
</body></html>
"""
    return md_text, html_text


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ca0", type=Path, default=None)
    p.add_argument("--cem-capture", type=Path, default=None)
    p.add_argument("--dump", type=Path, default=None)
    p.add_argument("--oracle-bank", type=Path, default=None)
    p.add_argument("--sweep", type=Path, default=None)
    p.add_argument("--cem-episodes", type=Path, default=None)
    p.add_argument("--ca1", type=Path, default=None, help="ignored (legacy)")
    p.add_argument("--pair", type=int, default=0)
    p.add_argument("--segment", type=int, default=0)
    p.add_argument("--tiers", default="1")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    tiers = parse_tiers(args.tiers)
    results, ca0_sum = collect_results(args, tiers)
    if not results:
        raise SystemExit("no figures produced; pass --ca0 / --cem-capture / --dump")
    th = load_thresholds()
    verdicts = compose_section_verdicts(results, th, ca0_summary=ca0_sum)
    bluf = compose_bluf(verdicts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md, html_doc = render_report(
        results,
        verdicts,
        _provenance(args, ca0_sum),
        tiers=tiers,
        fig_dir=out / "figures",
        bluf=bluf,
    )
    (out / "diagnostic_report.md").write_text(md)
    (out / "diagnostic_report.html").write_text(html_doc)
    (out / "verdicts.json").write_text(json.dumps({"bluf": bluf, "verdicts": verdicts}, indent=2))
    print(f"wrote {out / 'diagnostic_report.html'}")
    print(bluf)


if __name__ == "__main__":
    main()
