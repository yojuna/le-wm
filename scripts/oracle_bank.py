#!/usr/bin/env python3
"""C0.3-redo: collect live physics rollouts and window (t, t+25) short_horizon pairs.

  python scripts/oracle_bank.py --env pusht --controller goalpush \\
      --collect 60 --window 25 --seed 0 \\
      --out eval_results/pusht/c0_oracle_livebank/seed0/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

from eval_live import ENV_REGISTRY  # noqa: E402
from eval_logging.oracle_bank import (  # noqa: E402
    DEFAULT_WINDOW,
    save_oracle_bank,
    window_oracle_pairs,
)
from eval_logging.pairs import collect_trajectory_bank  # noqa: E402


def _controller_to_collector(name: str) -> str:
    n = name.strip().lower()
    if n in ("goalpush", "goal_push", "goal"):
        return "goal"
    if n in ("weak", "weak_policy"):
        return "weak"
    raise SystemExit(f"unknown --controller {name!r}; use goalpush or weak")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="pusht")
    p.add_argument("--controller", default="goalpush")
    p.add_argument("--collect", type=int, default=60, help="episodes to roll")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--stride", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--band", default="short_horizon")
    p.add_argument("--pack", default="tile_block")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-pairs", type=int, default=50)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--collect-steps", type=int, default=0)
    p.add_argument(
        "--weak-steps",
        type=int,
        default=20000,
        help="WeakPolicy steps if GoalPush undersupplies short_horizon windows",
    )
    args = p.parse_args(argv)
    if args.band != "short_horizon":
        raise SystemExit("only --band short_horizon is implemented")
    if args.pack != "tile_block":
        raise SystemExit("only --pack tile_block is implemented")

    spec = ENV_REGISTRY[args.env]
    collector = _controller_to_collector(args.controller)
    import stable_worldmodel as swm

    world = swm.World(
        env_name=spec.env_name,
        num_envs=1,
        max_episode_steps=spec.max_episode_steps,
        image_shape=(spec.img_size, spec.img_size),
        **spec.world_kwargs,
    )
    n_steps = args.collect_steps or max(args.collect * spec.max_episode_steps, 4000)
    print(f"collecting {collector} bank episodes={args.collect} steps={n_steps}")
    try:
        bank = collect_trajectory_bank(
            world,
            num_steps=n_steps,
            seed=args.seed,
            env_name=spec.env_name,
            min_episode_len=args.window + 1,
            collector=collector,
            num_episodes=None,
        )
    finally:
        world.close()

    print(
        f"  bank: {len(bank.episodes)} eps, collector={bank.collector}, "
        f"success={sum(1 for e in bank.episodes if e.succeeded)}"
    )
    all_pairs = window_oracle_pairs(
        bank,
        window=args.window,
        stride=args.stride,
        num_eval=None,
        seed=args.seed,
    )
    print(f"  short_horizon windows: {len(all_pairs)}")
    if len(all_pairs) < args.num_pairs:
        print(
            f"windows={len(all_pairs)} < {args.num_pairs}; collecting WeakPolicy "
            f"({args.weak_steps} steps)"
        )
        world = swm.World(
            env_name=spec.env_name,
            num_envs=1,
            max_episode_steps=spec.max_episode_steps,
            image_shape=(spec.img_size, spec.img_size),
            **spec.world_kwargs,
        )
        try:
            extra = collect_trajectory_bank(
                world,
                num_steps=args.weak_steps,
                seed=args.seed + 17,
                env_name=spec.env_name,
                min_episode_len=args.window + 1,
                collector="weak",
            )
        finally:
            world.close()
        bank.episodes.extend(extra.episodes)
        collector = f"{collector}+weak" if collector != "weak" else "weak"
        all_pairs = window_oracle_pairs(
            bank,
            window=args.window,
            stride=args.stride,
            num_eval=None,
            seed=args.seed,
        )
        print(
            f"  bank+weak: {len(bank.episodes)} eps, windows={len(all_pairs)}"
        )
    if len(all_pairs) < args.num_pairs:
        raise SystemExit(
            f"only {len(all_pairs)} short_horizon oracle windows, need {args.num_pairs}"
        )
    pairs = window_oracle_pairs(
        bank,
        window=args.window,
        stride=args.stride,
        num_eval=args.num_pairs,
        seed=args.seed,
    )

    out = Path(args.out)
    save_oracle_bank(
        out,
        pairs,
        extra={
            "oracle_source": collector,
            "controller": args.controller,
            "pair_band": "short_horizon",
            "action_pack": "tile_block",
            "window": args.window,
            "stride": args.stride,
            "scan_stride": 1,
            "seed": args.seed,
            "n_bank_episodes": len(bank.episodes),
        },
    )
    mean_pos = sum(p.pos_progress for p in pairs) / max(len(pairs), 1)
    print(
        f"wrote {out / 'pairs.npz'} n={len(pairs)} "
        f"window={args.window} mean_pos_progress={mean_pos:.1f} source={collector}"
    )


if __name__ == "__main__":
    main()
