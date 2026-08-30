#!/usr/bin/env python3
"""B.eval-block + B.eval-tercile: block-moving bank and frozen small-step frac.

Cuts frozen in thresholds.yaml (b_eval_block / b_eval_tercile) *before* this
run. Does not start Part B.

  python scripts/block_motion_eval.py \\
      --ca0 eval_results/pusht/ca0_closed_loop/seed0 \\
      --out eval_results/pusht/block_motion_eval/seed0 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

from encoder_floor import _pad_frames_actions, decide_guard  # noqa: E402
from eval_logging.oracle_bank import (  # noqa: E402
    pair_block_step_median,
    save_oracle_bank,
    window_block_moving_pairs,
)
from infidelity_confirm import (  # noqa: E402
    _encode_frames,
    _jsonable,
    _load_model,
    _m1_from_z,
    geometry_from_ca0,
)
from viz import load_ca0, load_thresholds  # noqa: E402


def frozen_tercile_edges(th: dict) -> tuple[float, float]:
    te = th.get("b_eval_tercile") or {}
    return float(te["adj_q33"]), float(te["adj_q67"])


def decide_block_eval(frac: float | None, small_frac: float | None, th: dict) -> dict:
    floor_th = th.get("encoder_floor") or th.get("infidelity_confirm") or {}
    accum = float(floor_th.get("frac_accum_at_or_below", 0.25))
    inf = float(floor_th.get("frac_infidelity_at_or_above", 0.5))
    if frac is None:
        overall = "CALIB_FAIL"
        reason = "No frac on the block-moving bank."
        gate_b = False
    elif frac <= accum:
        overall = "PUSHER_ONLY"
        reason = (
            "Block-moving frac ≤ 0.25. Live-bank infidelity was pusher hops. "
            "Part B is still earned but must not be certified on pusher-only frac."
        )
        gate_b = True
    elif frac >= inf:
        overall = "BLOCK_INFIDELITY"
        reason = "Block-moving windows are also infidelity. Claim strengthens to pusher and block."
        gate_b = True
    else:
        overall = "BLOCK_PARTIAL"
        reason = "Block-moving frac is between cuts. Retrain still gated; report honestly as tightening on block dynamics."
        gate_b = True
    small_note = None
    if small_frac is not None and frac is not None and small_frac > frac + 0.15:
        small_note = (
            f"Small-step tercile frac {small_frac:.2f} is worse than bank-median {frac:.2f}; "
            "B.eval-tercile remains load-bearing."
        )
    return {
        "overall": overall,
        "gate_part_b": gate_b,
        "reason": reason,
        "block_frac": frac,
        "small_step_frac": small_frac,
        "small_step_note": small_note,
        "cuts": {
            "frac_accum_at_or_below": accum,
            "frac_infidelity_at_or_above": inf,
            "median_step_block_xy_min": float((th.get("b_eval_block") or {}).get("median_step_block_xy_min", 2.0)),
            "adj_q33": float((th.get("b_eval_tercile") or {}).get("adj_q33")),
            "adj_q67": float((th.get("b_eval_tercile") or {}).get("adj_q67")),
        },
    }


def livebank_block_step_slice(ca0: dict, *, floor: float, edges: tuple[float, float]) -> dict:
    """Existing live-bank: parked vs moving *steps*, plus frozen terciles."""
    g = geometry_from_ca0(ca0, floor=floor, tercile_edges=edges)
    pose = g.get("pose") or {}
    return {
        "geometry_frozen_terciles": g,
        "block_xy_step_median": pose.get("block_xy_step_median"),
        "frac_on_block_moving_steps": pose.get("frac_on_block_moving_steps"),
        "frac_on_block_parked_steps": pose.get("frac_on_block_parked_steps"),
        "n_block_moving_steps": pose.get("n_block_moving_steps"),
        "n_block_parked_steps": pose.get("n_block_parked_steps"),
        "small_step_frac": (g.get("terciles") or {}).get("small", {}).get("frac"),
        "mid_step_frac": (g.get("terciles") or {}).get("mid", {}).get("frac"),
        "large_step_frac": (g.get("terciles") or {}).get("large", {}).get("frac"),
    }


def _collect_block_bank(args, th_block: dict) -> tuple[list, dict]:
    from eval_live import ENV_REGISTRY
    from eval_logging.pairs import collect_trajectory_bank
    import stable_worldmodel as swm

    spec = ENV_REGISTRY["pusht"]
    window = int(th_block.get("window", 25))
    stride = int(th_block.get("stride", 25))
    cut = float(th_block.get("median_step_block_xy_min", 2.0))
    need = int(th_block.get("n_pairs", 50))
    all_eps = []
    source_bits = []
    for batch, (collector, steps, seed) in enumerate(
        [
            ("goal", max(args.collect_episodes * spec.max_episode_steps, 8000), args.seed),
            ("weak", args.weak_steps, args.seed + 17),
            ("random", args.random_steps, args.seed + 31),
        ]
    ):
        world = swm.World(
            env_name=spec.env_name,
            num_envs=1,
            max_episode_steps=spec.max_episode_steps,
            image_shape=(spec.img_size, spec.img_size),
            **spec.world_kwargs,
        )
        print(f"collecting {collector} steps={steps} seed={seed}")
        try:
            bank = collect_trajectory_bank(
                world,
                num_steps=steps,
                seed=seed,
                env_name=spec.env_name,
                min_episode_len=window + 1,
                collector=collector,
                num_episodes=None,
            )
        finally:
            world.close()
        all_eps.extend(bank.episodes)
        source_bits.append(f"{collector}:{len(bank.episodes)}")
        from eval_logging.pairs import TrajectoryBank

        merged = TrajectoryBank(episodes=all_eps, env_name=spec.env_name, collector="+".join(source_bits))
        found = window_block_moving_pairs(
            merged,
            window=window,
            stride=stride,
            median_step_block_xy_min=cut,
            num_eval=None,
            seed=args.seed,
        )
        print(f"  after {collector}: episodes={len(all_eps)} block-moving windows={len(found)}")
        if len(found) >= need:
            break
    extra = 0
    from eval_logging.pairs import TrajectoryBank

    merged = TrajectoryBank(episodes=all_eps, env_name=spec.env_name, collector="+".join(source_bits))
    found = window_block_moving_pairs(
        merged, window=window, stride=stride, median_step_block_xy_min=cut, num_eval=None, seed=args.seed
    )
    while len(found) < need and extra < 4:
        extra += 1
        world = swm.World(
            env_name=spec.env_name,
            num_envs=1,
            max_episode_steps=spec.max_episode_steps,
            image_shape=(spec.img_size, spec.img_size),
            **spec.world_kwargs,
        )
        steps = 8000
        seed = args.seed + 100 + extra
        print(f"collecting extra goal batch {extra} steps={steps} seed={seed}")
        try:
            bank = collect_trajectory_bank(
                world,
                num_steps=steps,
                seed=seed,
                env_name=spec.env_name,
                min_episode_len=window + 1,
                collector="goal",
                num_episodes=None,
            )
        finally:
            world.close()
        all_eps.extend(bank.episodes)
        source_bits.append(f"goal_extra{extra}:{len(bank.episodes)}")
        merged = TrajectoryBank(episodes=all_eps, env_name=spec.env_name, collector="+".join(source_bits))
        found = window_block_moving_pairs(
            merged, window=window, stride=stride, median_step_block_xy_min=cut, num_eval=None, seed=args.seed
        )
        print(f"  extra {extra}: episodes={len(all_eps)} block-moving windows={len(found)}")
    from eval_logging.pairs import TrajectoryBank

    merged = TrajectoryBank(episodes=all_eps, env_name=spec.env_name, collector="+".join(source_bits))
    pairs = window_block_moving_pairs(
        merged,
        window=window,
        stride=stride,
        median_step_block_xy_min=cut,
        num_eval=need,
        seed=args.seed,
    )
    meta = {
        "oracle_source": "+".join(source_bits),
        "pair_band": "block_moving",
        "median_step_block_xy_min": cut,
        "window": window,
        "stride": stride,
        "n_bank_episodes": len(all_eps),
        "n_pairs": len(pairs),
        "seed": args.seed,
        "mean_block_step_median": float(np.mean([pair_block_step_median(p) for p in pairs])),
    }
    return pairs, meta


def _pairs_to_ca0(model, pairs, device, transform) -> dict:
    z_true_list, z_star_list, hat_list, state_list = [], [], [], []
    d_end = []
    for i, pair in enumerate(pairs):
        frames, acts, _ = _pad_frames_actions(
            pair.path_pixels if pair.path_pixels is not None else [pair.init_pixels],
            pair.oracle_actions,
        )
        z = _encode_frames(model, frames, transform, device)
        z_g = _encode_frames(model, [pair.goal_pixels], transform, device)[0]
        hat = _m1_from_z(model, z, acts, device)
        z_true_list.append(z)
        z_star_list.append(z_g)
        hat_list.append(hat)
        d_end.append(float(np.linalg.norm(hat[-1] - z_g)))
        st = np.asarray(pair.path_state, dtype=np.float32)
        if len(st) < len(frames):
            st = np.concatenate([st, np.repeat(st[-1:], len(frames) - len(st), axis=0)], axis=0)
        state_list.append(st[: len(frames)])
        if (i + 1) % 10 == 0:
            print(f"  block-bank pair {i + 1}/{len(pairs)}")
    z_true = np.stack(z_true_list, axis=0).astype(np.float32)
    z_hat = np.stack(hat_list, axis=0).astype(np.float32)
    return {
        "z_true": z_true,
        "z_star": np.stack(z_star_list, axis=0).astype(np.float32),
        "z_hat": z_hat[:, None],
        "m_values": np.array([1], dtype=np.int32),
        "d_end": np.asarray(d_end, dtype=np.float32)[:, None],
        "d_start": np.linalg.norm(z_true[:, 0] - np.stack(z_star_list), axis=-1).astype(np.float32),
        "path_state": np.stack(state_list, axis=0).astype(np.float32),
        "episode_id": np.arange(len(pairs), dtype=np.int32),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ca0", type=Path, required=True, help="seed-0 live-bank CA0 (frozen tercile source)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ckpt", default="hf_pusht")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--collect-episodes", type=int, default=80)
    p.add_argument("--weak-steps", type=int, default=20000)
    p.add_argument("--random-steps", type=int, default=8000)
    p.add_argument("--livebank-only", action="store_true", help="skip GPU collect; dump-only slices")
    args = p.parse_args(argv)
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    th = load_thresholds()
    edges = frozen_tercile_edges(th)
    ca0, _ = load_ca0(args.ca0)
    live = livebank_block_step_slice(ca0, floor=0.0, edges=edges)
    (args.out / "livebank_frozen_terciles.json").write_text(json.dumps(live, indent=2, default=str))
    print("live-bank frozen terciles / block-step slices")
    print(json.dumps({k: live[k] for k in live if k != "geometry_frozen_terciles"}, indent=2, default=str))

    if args.livebank_only:
        decision = decide_block_eval(None, live.get("small_step_frac"), th)
        (args.out / "summary.json").write_text(
            json.dumps({"decision": decision, "livebank": _jsonable(live), "block_bank": None}, indent=2, default=str)
        )
        print(json.dumps(decision, indent=2))
        return

    bank_dir = args.out / "block_moving_bank"
    from eval_logging.oracle_bank import load_oracle_bank

    if (bank_dir / "pairs.npz").exists():
        pairs, meta = load_oracle_bank(bank_dir)
        print(f"reusing {bank_dir} n={len(pairs)}")
    else:
        pairs, meta = _collect_block_bank(args, th.get("b_eval_block") or {})
        save_oracle_bank(bank_dir, pairs, extra=meta)
        print(f"wrote {bank_dir} n={len(pairs)} mean_block_step_median={meta.get('mean_block_step_median')}")

    model, device, transform = _load_model(args.ckpt, args.device)
    print("encoding block-moving bank m=1")
    ca0_b = _pairs_to_ca0(model, pairs, device, transform)
    np.savez_compressed(args.out / "ca0_block.npz", **ca0_b)
    geom = geometry_from_ca0(ca0_b, floor=0.0, tercile_edges=edges)
    small_frac = (geom.get("terciles") or {}).get("small", {}).get("frac")
    decision = decide_block_eval(geom.get("frac"), small_frac, th)
    payload = {
        "decision": decision,
        "livebank": _jsonable(live),
        "block_bank": {
            "meta": meta,
            "geometry": geom,
            "n_pairs": len(pairs),
            "mean_pair_block_step_median": float(np.mean([pair_block_step_median(p) for p in pairs])),
            "guard": decide_guard(geom.get("frac"), calib_ok=True, th=th.get("encoder_floor") or {}),
        },
    }
    (args.out / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2, default=str))
    print(json.dumps({"decision": decision, "block_frac": geom.get("frac"), "angle": geom.get("mean_angle_deg"), "pose": geom.get("pose"), "terciles": geom.get("terciles")}, indent=2, default=str))
    print(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
