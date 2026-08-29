#!/usr/bin/env python3
"""B2: per-step drift curves and action-shuffle liveness from a Phase B dump.

  python scripts/predictor_drift.py --dump eval_results/pusht/phase_b_dump/seed0/dump.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase_b import CEM_HORIZON, HISTORY, load_dump, summarize_drift  # noqa: E402


def plot_drift(mean_true: np.ndarray, mean_shuf: np.ndarray, out: Path) -> None:
    h = np.arange(len(mean_true))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h, mean_true, marker="o", label="true actions")
    ax.plot(h, mean_shuf, marker="s", label="shuffled future actions")
    cem_idx = HISTORY + CEM_HORIZON - 1
    if cem_idx < len(mean_true):
        ax.axvline(cem_idx, color="k", ls="--", lw=1, label=f"CEM h={CEM_HORIZON} (index {cem_idx})")
    ax.axvline(HISTORY - 1, color="gray", ls=":", lw=1, label="end of teacher-forced history")
    ax.set_xlabel("frame index along segment (0 = start)")
    ax.set_ylabel("mean ‖ẑ − z‖₂")
    ax.set_title("Predictor drift vs horizon (true-action vs shuffled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    dump = load_dump(args.dump)
    meta = dump.get("meta") or {}
    drift = dump["drift_true"]  # (N, L)
    shuf = dump["drift_shuf"]
    mean_true = drift.mean(axis=0)
    mean_shuf = shuf.mean(axis=0)
    stats = {
        "dump": str(args.dump),
        "env": meta.get("env"),
        "n_segments": int(drift.shape[0]),
        "segment_len": int(drift.shape[1]),
        "true": summarize_drift(mean_true),
        "shuffled": summarize_drift(mean_shuf),
        "liveness_mean_gap": float((mean_shuf - mean_true)[HISTORY:].mean()),
        "liveness_end_gap": float(mean_shuf[-1] - mean_true[-1]),
        "note": (
            "mean_all_frames includes teacher-forced zeros. "
            "Use mean_predicted_only and at_h5_index for CEM. "
            "Positive liveness gap: shuffled actions drift more than true actions."
        ),
    }
    out_dir = Path(args.out_dir or args.dump.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_drift(mean_true, mean_shuf, out_dir / "drift_curve.png")
    (out_dir / "drift_summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in ("env", "true", "shuffled", "liveness_end_gap")}, indent=2))
    print(f"wrote {out_dir / 'drift_summary.json'}")


if __name__ == "__main__":
    main()
