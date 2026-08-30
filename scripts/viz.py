#!/usr/bin/env python3
"""CLI for dump-driven lewm-phi figures (spec 15).

  python scripts/viz.py --ca0 eval_results/pusht/ca0_closed_loop/seed0/ \\
      --figs oracle_overlay --pair 0 --out eval_results/pusht/viz/seed0/

  python scripts/viz.py --dump eval_results/pusht/phase_b_dump/seed0/dump.npz \\
      --figs drift_contacts,rank_spectrum,probe_faithfulness
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_logging.oracle_bank import load_oracle_bank  # noqa: E402
from viz import (  # noqa: E402
    FIG_REGISTRY,
    fig_cem_landscape,
    fig_drift_contacts,
    fig_intervention_sweep,
    fig_oracle_overlay,
    fig_probe_faithfulness,
    fig_rank_spectrum,
    fig_rollout_filmstrip,
    load_ca0,
    load_cem_capture,
    load_dump,
    save_figure,
)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, default=None)
    p.add_argument("--ca0", type=Path, default=None)
    p.add_argument("--oracle-bank", type=Path, default=None)
    p.add_argument("--cem-capture", type=Path, default=None)
    p.add_argument("--sweep", type=Path, default=None, help="ca3 sweep.json")
    p.add_argument("--figs", default="oracle_overlay")
    p.add_argument("--pair", type=int, default=0)
    p.add_argument("--segment", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    names = [n.strip() for n in args.figs.split(",") if n.strip()]
    unknown = [n for n in names if n not in FIG_REGISTRY]
    if unknown:
        raise SystemExit(f"unknown figs {unknown}; choose from {sorted(FIG_REGISTRY)}")

    out = args.out
    if out is None:
        if args.ca0:
            out = Path(args.ca0) / "figs"
        elif args.dump:
            out = Path(args.dump).parent / "figs"
        else:
            out = Path("eval_results/pusht/viz")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    dump = load_dump(args.dump) if args.dump else None
    ca0, ca0_sum = load_ca0(args.ca0) if args.ca0 else (None, {})
    captions = {}

    for name in names:
        if name == "oracle_overlay":
            if ca0 is None:
                raise SystemExit("oracle_overlay needs --ca0")
            fig = fig_oracle_overlay(ca0, pair=args.pair, summary=ca0_sum)
        elif name == "cem_landscape":
            if args.cem_capture is None:
                raise SystemExit("cem_landscape needs --cem-capture")
            fig = fig_cem_landscape(load_cem_capture(args.cem_capture))
        elif name == "drift_contacts":
            src = dump if dump is not None else ca0
            if src is None:
                raise SystemExit("drift_contacts needs --dump or --ca0")
            fig = fig_drift_contacts(src, segment=args.segment)
        elif name == "rank_spectrum":
            if dump is None:
                raise SystemExit("rank_spectrum needs --dump")
            fig = fig_rank_spectrum(dump)
        elif name == "probe_faithfulness":
            if dump is None:
                raise SystemExit("probe_faithfulness needs --dump")
            fig = fig_probe_faithfulness(dump)
        elif name == "intervention_sweep":
            if args.sweep is None:
                raise SystemExit("intervention_sweep needs --sweep JSON")
            fig = fig_intervention_sweep(json.loads(Path(args.sweep).read_text()))
        elif name == "rollout_filmstrip":
            if ca0 is None or args.oracle_bank is None:
                raise SystemExit("rollout_filmstrip needs --ca0 and --oracle-bank")
            pairs, _ = load_oracle_bank(args.oracle_bank)
            pair = pairs[args.pair]
            m_values = [int(x) for x in ca0["m_values"]]
            mi = m_values.index(25) if 25 in m_values else int(max(range(len(m_values)), key=lambda i: m_values[i]))
            fig = fig_rollout_filmstrip(
                pair.path_pixels,
                ca0["z_true"][args.pair],
                ca0["z_hat"][args.pair, mi],
            )
        else:
            raise SystemExit(name)
        cap = {
            "motivates": getattr(fig, "_viz_motivates", None),
            "scalars": getattr(fig, "_viz_scalars", None),
            "captured_variance": getattr(fig, "_viz_captured_variance", None),
        }
        stem = out / name
        save_figure(fig, stem)
        cap["png"] = str(stem.with_suffix(".png"))
        cap["pdf"] = str(stem.with_suffix(".pdf"))
        captions[name] = cap
        print(f"wrote {stem.with_suffix('.png')}")
        if name == "rank_spectrum" and cap.get("scalars"):
            (out / "rank_spectrum.json").write_text(
                json.dumps(_jsonable(cap["scalars"]), indent=2)
            )
    (out / "captions.json").write_text(
        json.dumps(_jsonable(captions), indent=2, default=str)
    )


if __name__ == "__main__":
    main()
