#!/usr/bin/env python3
"""CLI for dump-driven lewm-phi figures (spec 15 v3).

  python scripts/viz.py --ca0 eval_results/pusht/ca0_closed_loop/seed0/ \\
      --figs a1,a2 --pair 0 --out eval_results/pusht/viz/seed0/
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
    FigureResult,
    as_figure,
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


def _emit(result, name: str, out: Path, captions: dict) -> None:
    fig = as_figure(result)
    cap = {
        "motivates": getattr(result, "motivates", None) or getattr(fig, "_viz_motivates", None),
        "scalars": result.scalars if isinstance(result, FigureResult) else getattr(fig, "_viz_scalars", None),
        "question": getattr(result, "question", None),
        "tier": getattr(result, "tier", None),
        "caption": getattr(result, "caption", None),
        "captured_variance": getattr(fig, "_viz_captured_variance", None),
    }
    stem = out / name
    save_figure(fig, stem)
    cap["png"] = str(stem.with_suffix(".png"))
    cap["pdf"] = str(stem.with_suffix(".pdf"))
    captions[name] = cap
    print(f"wrote {stem.with_suffix('.png')}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, default=None)
    p.add_argument("--ca0", type=Path, default=None)
    p.add_argument("--oracle-bank", type=Path, default=None)
    p.add_argument("--cem-capture", type=Path, default=None)
    p.add_argument("--sweep", type=Path, default=None)
    p.add_argument("--cem-episodes", type=Path, default=None, help="episodes.csv for e1")
    p.add_argument("--figs", default="a1,a2")
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
        if name in ("a1", "oracle_overlay"):
            if ca0 is None:
                raise SystemExit(f"{name} needs --ca0")
            result = fig_a1(ca0, pair=args.pair, summary=ca0_sum)
        elif name == "a2":
            if ca0 is None:
                raise SystemExit("a2 needs --ca0")
            result = fig_a2(ca0, summary=ca0_sum)
        elif name in ("b1", "cem_landscape"):
            if args.cem_capture is None:
                raise SystemExit(f"{name} needs --cem-capture")
            result = fig_b1(load_cem_capture(args.cem_capture))
        elif name == "b2":
            if args.cem_capture is None:
                raise SystemExit("b2 needs --cem-capture")
            result = fig_b2(load_cem_capture(args.cem_capture))
        elif name == "b3":
            if args.cem_capture is None:
                raise SystemExit("b3 needs --cem-capture")
            result = fig_b3(load_cem_capture(args.cem_capture))
        elif name in ("a3", "drift_contacts"):
            src = dump if dump is not None else ca0
            if src is None:
                raise SystemExit(f"{name} needs --dump or --ca0")
            result = fig_a3(src, segment=args.segment)
        elif name == "a4":
            if ca0 is None or args.oracle_bank is None:
                raise SystemExit("a4 needs --ca0 and --oracle-bank")
            pairs, _ = load_oracle_bank(args.oracle_bank)
            import numpy as np

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
            result = fig_a4(ca0, stacked)
        elif name == "a5":
            if ca0 is None:
                raise SystemExit("a5 needs --ca0")
            result = fig_a5(ca0, pair=args.pair)
        elif name in ("d1", "rank_spectrum"):
            if dump is None:
                raise SystemExit(f"{name} needs --dump")
            result = fig_d1(dump)
        elif name in ("c2", "probe_faithfulness"):
            if dump is None:
                raise SystemExit(f"{name} needs --dump")
            result = fig_c2(dump)
        elif name in ("d2", "intervention_sweep"):
            if args.sweep is None:
                raise SystemExit(f"{name} needs --sweep JSON")
            result = fig_d2(json.loads(Path(args.sweep).read_text()))
        elif name in ("f1", "rollout_filmstrip"):
            if ca0 is None or args.oracle_bank is None:
                raise SystemExit(f"{name} needs --ca0 and --oracle-bank")
            pairs, _ = load_oracle_bank(args.oracle_bank)
            pair = pairs[args.pair]
            m_values = [int(x) for x in ca0["m_values"]]
            mi = m_values.index(25) if 25 in m_values else int(max(range(len(m_values)), key=lambda i: m_values[i]))
            result = fig_f1(
                pair.path_pixels,
                ca0["z_true"][args.pair],
                ca0["z_hat"][args.pair, mi],
            )
        elif name == "e1":
            if args.cem_episodes is None or args.oracle_bank is None:
                raise SystemExit("e1 needs --cem-episodes and --oracle-bank")
            import numpy as np

            pairs, _ = load_oracle_bank(args.oracle_bank)
            import csv

            rows = list(csv.DictReader(Path(args.cem_episodes).open()))
            n = min(len(pairs), len(rows))
            xy = []
            suc = []
            for i in range(n):
                st = np.asarray(pairs[i].path_state if pairs[i].path_state is not None else pairs[i].goal_state)
                if st.ndim == 1:
                    xy.append(st[2:4])
                else:
                    xy.append(st[-1, 2:4])
                suc.append(str(rows[i].get("success", "")).lower() in ("true", "1", "yes"))
            result = fig_e1(goal_xy=np.stack(xy), success=np.asarray(suc))
        else:
            raise SystemExit(name)
        _emit(result, name, out, captions)
    (out / "captions.json").write_text(json.dumps(_jsonable(captions), indent=2, default=str))


if __name__ == "__main__":
    main()
