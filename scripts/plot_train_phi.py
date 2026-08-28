#!/usr/bin/env python3
"""Plot train_phi curves from an existing train_phi_meta.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_phi import plot_training_curves, _append_metrics_csv  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="path to train_phi_meta.json",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="defaults to meta parent dir",
    )
    args = p.parse_args()

    meta_path = args.meta
    if meta_path is None:
        import os

        cache = Path(os.environ.get("STABLEWM_HOME", ROOT.parent / "stablewm"))
        meta_path = cache / "checkpoints" / "pusht" / "lewm_phi" / "train_phi_meta.json"
    meta = json.loads(meta_path.read_text())
    out_dir = args.out_dir or meta_path.parent
    history = meta["history"]
    csv_path = out_dir / "metrics.csv"
    if csv_path.exists():
        csv_path.unlink()
    for row in history:
        _append_metrics_csv(csv_path, row)
    png = out_dir / "training_curves.png"
    plot_training_curves(history, png)
    print(f"wrote {png}")
    print(f"wrote {csv_path}")
    print(f"best_corr={meta.get('best_corr')}")


if __name__ == "__main__":
    main()
