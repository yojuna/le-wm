#!/usr/bin/env python3
"""Shim: gallery stitcher replaced by scripts/report.py (spec 15 v3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report import main as report_main  # noqa: E402


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
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
    p.add_argument("--cem-capture", type=Path, default=None)
    p.add_argument("--dump", type=Path, default=None)
    p.add_argument("--oracle-bank", type=Path, default=None)
    p.add_argument("--cem-episodes", type=Path, default=None)
    p.add_argument("--tiers", default="1")
    args = p.parse_args(argv)
    fwd = ["--out", str(args.out), "--tiers", args.tiers]
    if args.ca0:
        fwd += ["--ca0", str(args.ca0)]
    if args.cem_capture:
        fwd += ["--cem-capture", str(args.cem_capture)]
    if args.dump:
        fwd += ["--dump", str(args.dump)]
    if args.oracle_bank:
        fwd += ["--oracle-bank", str(args.oracle_bank)]
    if args.sweep:
        fwd += ["--sweep", str(args.sweep)]
    if args.cem_episodes:
        fwd += ["--cem-episodes", str(args.cem_episodes)]
    if args.ca1:
        fwd += ["--ca1", str(args.ca1)]
    report_main(fwd)


if __name__ == "__main__":
    main()
