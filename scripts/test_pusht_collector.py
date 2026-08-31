#!/usr/bin/env python3
"""Test A: kinematic collector yields successful eps + progress pairs.

    python scripts/test_pusht_collector.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))
os.environ.setdefault("MUJOCO_GL", "egl")


def main() -> int:
    import stable_worldmodel as swm
    from eval_logging.pairs import (
        bank_success_report,
        collect_trajectory_bank,
        sample_eval_pairs,
    )

    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=1,
        max_episode_steps=100,
        image_shape=(224, 224),
    )
    try:
        print("collecting kinematic bank (48 episodes × 80 steps)...")
        bank = collect_trajectory_bank(
            world,
            num_steps=48 * 80,
            seed=0,
            env_name="swm/PushT-v1",
            collector="kinematic",
            num_episodes=48,
            kinematic_horizon=80,
        )
        report = bank_success_report(bank)
        print(
            f"collector={report['collector']} episodes={report['episodes']} "
            f"success={report['success_episodes']} "
            f"({report['success_rate_pct']:.1f}%) "
            f"mean_len={report['mean_length']:.1f}"
        )
        if report["success_rate_pct"] < 90:
            print("FAIL: kinematic episodes should almost always succeed")
            return 1

        pairs = sample_eval_pairs(
            bank, num_eval=16, goal_offset=25, seed=0, min_pos_delta=15.0
        )
        n_from_ok = sum(1 for p in pairs if p.from_success_ep)
        mean_prog = sum(p.pos_progress for p in pairs) / len(pairs)
        print(
            f"sampled {len(pairs)} pairs; from_success_ep={n_from_ok}; "
            f"mean_pos_progress={mean_prog:.1f}"
        )
        if n_from_ok < 8 or mean_prog < 15:
            print("FAIL: expected success-sourced progress pairs")
            return 1
        print("PASS kinematic collector")
        return 0
    finally:
        world.close()


if __name__ == "__main__":
    raise SystemExit(main())
