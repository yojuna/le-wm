#!/usr/bin/env python3
"""CA1: free-space vs contact mean drift scalars (no figures).

  python scripts/drift_by_event.py \\
      --dump eval_results/pusht/phase_b_dump/seed0/dump.npz \\
      --out eval_results/pusht/ca1_drift_contacts/seed0/

  python scripts/drift_by_event.py \\
      --ca0 eval_results/pusht/ca0_closed_loop/seed0/ \\
      --out eval_results/pusht/ca1_drift_contacts/seed0/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase_b import HISTORY, contact_events, load_dump  # noqa: E402


def _stats(drift: np.ndarray, mask: np.ndarray) -> dict:
    pred = np.zeros_like(mask, dtype=bool)
    if drift.ndim == 2 and drift.shape[1] > HISTORY:
        pred[:, HISTORY:] = True
    use = mask & pred if pred.any() else mask
    vals = drift[use]
    return {
        "n_steps": int(use.sum()),
        "mean_drift": None if not vals.size else float(vals.mean()),
        "median_drift": None if not vals.size else float(np.median(vals)),
    }


def _ratio(drift: np.ndarray, contact_mask: np.ndarray, free_mask: np.ndarray):
    c = _stats(drift, contact_mask)["mean_drift"]
    f = _stats(drift, free_mask)["mean_drift"]
    if c is None or f is None:
        return None
    return float(c / max(f, 1e-8))


def from_dump(dump: dict, env: str) -> dict:
    drift = np.asarray(dump["drift_true"], dtype=np.float64)
    state = np.asarray(dump["state"])
    ev = contact_events(state, env=env)
    free = ~ev["any"]
    return {
        "source": "dump",
        "env": env,
        "n_segments": int(drift.shape[0]),
        "segment_len": int(drift.shape[1]),
        "contact": _stats(drift, ev["contact"]),
        "wall": _stats(drift, ev["wall"]),
        "any_event": _stats(drift, ev["any"]),
        "free": _stats(drift, free),
        "ratio_contact_over_free": _ratio(drift, ev["contact"], free),
    }


def from_ca0(ca0_dir: Path, env: str) -> dict:
    blob = np.load(ca0_dir / "ca0.npz", allow_pickle=True)
    z_true = blob["z_true"]
    z_hat = blob["z_hat"]
    m_values = [int(x) for x in blob["m_values"]]
    try:
        m25 = m_values.index(25)
    except ValueError:
        m25 = int(np.argmax(m_values))
    hat = z_hat[:, m25]
    drift = np.linalg.norm(hat - z_true, axis=-1)
    state = blob["path_state"]
    ev = contact_events(state, env=env)
    free = ~ev["any"]
    return {
        "source": "ca0",
        "env": env,
        "m_open_loop": int(m_values[m25]),
        "n_pairs": int(z_true.shape[0]),
        "contact": _stats(drift, ev["contact"]),
        "wall": _stats(drift, ev["wall"]),
        "any_event": _stats(drift, ev["any"]),
        "free": _stats(drift, free),
        "ratio_contact_over_free": _ratio(drift, ev["contact"], free),
        "note": "Uses CA0 m=25 (or largest m) ẑ vs encoded true path.",
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, action="append", default=[])
    p.add_argument("--ca0", type=Path, default=None)
    p.add_argument("--env", default="pusht")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if not args.dump and not args.ca0:
        raise SystemExit("pass --dump and/or --ca0")

    rows = []
    for dump_path in args.dump:
        dump = load_dump(dump_path)
        env = str((dump.get("meta") or {}).get("env") or args.env)
        row = from_dump(dump, env)
        row["dump"] = str(dump_path)
        rows.append(row)
    if args.ca0:
        rows.append(from_ca0(Path(args.ca0), args.env))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": rows,
        "note": (
            "Scalars only. Fig-3 in viz.py draws the curve. "
            "ratio_contact_over_free >> 1 suggests a contact-representation fault."
        ),
    }
    path = out / "drift_by_event.json"
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
