#!/usr/bin/env python3
"""Persist z, ẑ, state factors, k, pose error for Phase B probes/drift.

Does not assume offset_autopsy artifacts. Writes dump.npz + dump.meta.json.

  python scripts/latent_dump.py --env pusht --seed 0 --device cuda
  python scripts/latent_dump.py --env reacher --seed 0 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

from eval_live import ENV_REGISTRY, cache_dir, download_spec  # noqa: E402
from eval_logging.pairs import collect_trajectory_bank  # noqa: E402
from eval_setup import load_lewm_checkpoint  # noqa: E402
from phase_b import (  # noqa: E402
    CEM_HORIZON,
    DUMP_VERSION,
    HISTORY,
    encode_episode_frames,
    factor_names_for_env,
    imagine_path,
    img_transform,
    per_step_drift,
    pad_action,
    remaining_pose_error,
    save_dump,
    shuffle_future_actions,
    summarize_drift,
)


def _require_cuda(device: str, allow_cpu: bool) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        if not allow_cpu:
            raise SystemExit("CUDA required (pass --allow-cpu to override)")
        print("CUDA not available; using cpu")
        return torch.device("cpu")
    return torch.device(device)


def _state_block(ep, start: int, length: int) -> np.ndarray:
    if not ep.state or start + length > len(ep.state):
        dim = len(ep.state[0]) if ep.state else 1
        return np.zeros((length, dim), dtype=np.float32)
    return np.stack(
        [np.asarray(ep.state[t], dtype=np.float32).reshape(-1) for t in range(start, start + length)],
        axis=0,
    )


def sample_segments(
    bank,
    *,
    n_segments: int,
    segment_len: int,
    seed: int,
    min_length: int,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    usable = [
        (i, ep)
        for i, ep in enumerate(bank.episodes)
        if len(ep) >= min_length and len(ep.pixels) >= min_length
    ]
    if not usable:
        raise RuntimeError(
            f"no episodes with length >= {min_length} "
            f"(bank has {len(bank.episodes)} eps)"
        )
    out: list[tuple[int, int]] = []
    for _ in range(n_segments):
        ep_i, ep = usable[int(rng.integers(0, len(usable)))]
        max_start = len(ep) - segment_len
        start = int(rng.integers(0, max_start + 1))
        out.append((ep_i, start))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=["pusht", "reacher"], default="pusht")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--collect-episodes", type=int, default=64)
    p.add_argument("--collect-steps", type=int, default=4000)
    p.add_argument("--n-segments", type=int, default=48)
    p.add_argument("--segment-len", type=int, default=26, help="GOAL_OFFSET+1 default")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    spec = ENV_REGISTRY[args.env]
    device = _require_cuda(args.device, args.allow_cpu)
    weights = cache_dir() / "checkpoints" / spec.ckpt_dir / "weights.pt"
    if not weights.exists() or weights.stat().st_size < 50_000_000:
        print(f"checkpoint missing at {weights}; downloading")
        download_spec(spec)

    import stable_worldmodel as swm

    world = swm.World(
        env_name=spec.env_name,
        num_envs=1,
        max_episode_steps=spec.max_episode_steps,
        image_shape=(spec.img_size, spec.img_size),
        **spec.world_kwargs,
    )
    if args.env == "pusht":
        bank = collect_trajectory_bank(
            world,
            num_steps=args.collect_steps,
            seed=args.seed,
            env_name=spec.env_name,
            min_episode_len=args.segment_len,
            collector="kinematic",
            num_episodes=args.collect_episodes,
            kinematic_horizon=80,
        )
    else:
        bank = collect_trajectory_bank(
            world,
            num_steps=args.collect_steps,
            seed=args.seed,
            env_name=spec.env_name,
            min_episode_len=args.segment_len,
            collector="weak",  # non-PushT → random actions
        )
    world.close()

    model = load_lewm_checkpoint(spec.ckpt_dir)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    transform = img_transform(args.img_size)

    segs = sample_segments(
        bank,
        n_segments=args.n_segments,
        segment_len=args.segment_len,
        seed=args.seed,
        min_length=args.segment_len,
    )
    rng = np.random.default_rng(args.seed + 1)

    z_list, hat_list, shuf_list = [], [], []
    state_list, k_list, pose_list = [], [], []
    ep_ids, starts = [], []
    drift_rows, shuf_drift_rows = [], []

    for ep_i, start in segs:
        ep = bank.episodes[ep_i]
        L = args.segment_len
        z = encode_episode_frames(model, ep, start, L, transform, device)
        acts = np.stack(
            [
                pad_action(ep.action[t] if t < len(ep.action) else np.zeros(1))
                for t in range(start, start + L)
            ],
            axis=0,
        )
        z_hat = imagine_path(model, z, acts, device=device)
        acts_shuf = shuffle_future_actions(acts, HISTORY, rng)
        z_shuf = imagine_path(model, z, acts_shuf, device=device)
        st = _state_block(ep, start, L)
        rem_k = np.arange(L - 1, -1, -1, dtype=np.float32)
        rem_pose = remaining_pose_error(args.env, st).astype(np.float32)

        z_list.append(z.numpy())
        hat_list.append(z_hat.numpy())
        shuf_list.append(z_shuf.numpy())
        state_list.append(st)
        k_list.append(rem_k)
        pose_list.append(rem_pose)
        ep_ids.append(ep_i)
        starts.append(start)
        drift_rows.append(per_step_drift(z, z_hat))
        shuf_drift_rows.append(per_step_drift(z, z_shuf))

    z_arr = np.stack(z_list, axis=0)
    state_arr = np.stack(state_list, axis=0)
    names = factor_names_for_env(args.env, state_arr.shape[-1])
    drift = np.stack(drift_rows, axis=0)
    shuf_drift = np.stack(shuf_drift_rows, axis=0)
    mean_drift = drift.mean(axis=0)
    mean_shuf = shuf_drift.mean(axis=0)

    out_dir = args.out_dir or (
        ROOT / "eval_results" / args.env / "phase_b_dump" / f"seed{args.seed}"
    )
    dump_path = Path(out_dir) / "dump.npz"
    meta = {
        "version": DUMP_VERSION,
        "env": args.env,
        "seed": args.seed,
        "ckpt": spec.ckpt_dir,
        "n_segments": int(z_arr.shape[0]),
        "segment_len": int(z_arr.shape[1]),
        "z_dim": int(z_arr.shape[2]),
        "history": HISTORY,
        "cem_horizon": CEM_HORIZON,
        "factor_names": list(names),
        "collector": bank.collector,
        "n_bank_episodes": len(bank.episodes),
        "drift_true": summarize_drift(mean_drift),
        "drift_shuffled": summarize_drift(mean_shuf),
        "action_liveness_end_gap": float(mean_shuf[-1] - mean_drift[-1]),
    }
    save_dump(
        dump_path,
        {
            "z": z_arr.astype(np.float32),
            "z_hat": np.stack(hat_list, axis=0).astype(np.float32),
            "z_hat_shuf": np.stack(shuf_list, axis=0).astype(np.float32),
            "state": state_arr.astype(np.float32),
            "remaining_k": np.stack(k_list, axis=0),
            "remaining_pose": np.stack(pose_list, axis=0),
            "episode_id": np.asarray(ep_ids, dtype=np.int32),
            "start": np.asarray(starts, dtype=np.int32),
            "drift_true": drift.astype(np.float32),
            "drift_shuf": shuf_drift.astype(np.float32),
            "factor_names": np.asarray(names),
            "meta": meta,
        },
    )
    print(f"wrote {dump_path}")
    print(
        "drift mean_all={mean_all_frames:.3f} predicted_only={mean_predicted_only:.3f} "
        "at_h5={at_h5_index:.3f} end={end:.3f}".format(**meta["drift_true"])
    )
    print(f"action liveness end gap (shuf - true)={meta['action_liveness_end_gap']:.3f}")


if __name__ == "__main__":
    main()
