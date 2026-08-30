#!/usr/bin/env python3
"""CA0: closed-loop imagine discriminator (re-encode every m steps).

Persists trajectories for Fig-1. No matplotlib — figures live in viz.py.

  python scripts/closed_loop_imagine.py \\
      --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0/ \\
      --reencode-every 1 3 5 12 25 --pack tile_block --device cuda \\
      --out eval_results/pusht/ca0_closed_loop/seed0/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

from eval_live import ENV_REGISTRY  # noqa: E402
from eval_logging.oracle_bank import load_oracle_bank  # noqa: E402
from eval_setup import load_lewm_checkpoint  # noqa: E402
from phase_b import (  # noqa: E402
    HISTORY,
    encode_frames,
    imagine_closed_loop,
    img_transform,
)
from phi_data import frame_to_tensor  # noqa: E402

# Pre-registered in docs/14_phase_c_alt_plan.md (do not retune after seeing numbers).
ACCUM_TOWARD_AT_M5 = 0.60
ACCUM_D_END_AT_M5 = 3.0
M1_TOWARD_GUARD = 0.90
M1_D_END_GUARD = 1.0
INFIDELITY_TOWARD_AT_M3 = 0.60
INFIDELITY_D_END_AT_M3 = 5.0


@torch.no_grad()
def _encode_frame(model, frame, transform, device) -> torch.Tensor:
    pix = frame_to_tensor(frame, transform).unsqueeze(0).to(device)
    return encode_frames(model, pix)[0].cpu()


def _pad_frames_actions(frames, acts):
    frames = [np.asarray(fr) for fr in frames]
    if len(frames) < HISTORY + 1:
        while len(frames) < HISTORY + 1:
            frames.append(frames[0])
    L = len(frames)
    acts = np.asarray(acts, dtype=np.float32)
    if acts.ndim == 1:
        acts = acts.reshape(1, -1)
    if len(acts) < L:
        pad = np.zeros((L - len(acts), acts.shape[-1] if acts.size else 2), dtype=np.float32)
        acts = np.concatenate([acts, pad], axis=0) if acts.size else pad
    elif len(acts) > L:
        acts = acts[:L]
    return frames, acts, L


def decide_fork(by_m: dict[int, dict]) -> dict:
    """Apply pre-registered CA0 thresholds. Pictures are not the gate."""
    m1 = by_m.get(1)
    m3 = by_m.get(3)
    m5 = by_m.get(5)
    reason = []
    if m1 is not None:
        if m1["frac_toward"] < M1_TOWARD_GUARD or m1["mean_d_end"] > M1_D_END_GUARD:
            reason.append(
                f"m=1 not near-perfect (toward={m1['frac_toward']:.3f}, "
                f"d_end={m1['mean_d_end']:.3f})"
            )
            return {
                "fork": "CA0-INFIDELITY",
                "reason": " ".join(reason) + " — single-step prediction fails the guard.",
            }
    if m5 is not None:
        if (
            m5["frac_toward"] >= ACCUM_TOWARD_AT_M5
            and m5["mean_d_end"] <= ACCUM_D_END_AT_M5
        ):
            return {
                "fork": "CA0-ACCUMULATION",
                "reason": (
                    f"m=5 toward={m5['frac_toward']:.3f} (>= {ACCUM_TOWARD_AT_M5}) "
                    f"and d_end={m5['mean_d_end']:.3f} (<= {ACCUM_D_END_AT_M5})"
                ),
            }
    if m3 is not None:
        if (
            m3["frac_toward"] < INFIDELITY_TOWARD_AT_M3
            or m3["mean_d_end"] > INFIDELITY_D_END_AT_M3
        ):
            return {
                "fork": "CA0-INFIDELITY",
                "reason": (
                    f"drift persists at m=3 (toward={m3['frac_toward']:.3f}, "
                    f"d_end={m3['mean_d_end']:.3f})"
                ),
            }
    return {
        "fork": "CA0-AMBIGUOUS",
        "reason": "m=5 did not meet ACCUMULATION cuts; m=3 did not meet INFIDELITY cuts.",
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle-bank", type=Path, required=True)
    p.add_argument("--env", default="pusht")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pack", default="tile_block")
    p.add_argument(
        "--reencode-every",
        type=int,
        nargs="+",
        default=[1, 3, 5, 12, 25],
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    if args.pack != "tile_block":
        raise SystemExit("only --pack tile_block is implemented")

    spec = ENV_REGISTRY[args.env]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_lewm_checkpoint(spec.ckpt_dir)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    transform = img_transform(spec.img_size)

    pairs, bank_meta = load_oracle_bank(args.oracle_bank)
    m_values = [int(x) for x in args.reencode_every]
    out_dir = Path(args.out or (Path(args.oracle_bank) / ".." / "ca0_closed_loop" / "seed0"))
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    z_true_list = []
    z_star_list = []
    path_state_list = []
    d_start_list = []
    hats = {m: [] for m in m_values}
    d_end = {m: [] for m in m_values}
    toward = {m: [] for m in m_values}

    for i, pair in enumerate(pairs):
        frames, acts, _L = _pad_frames_actions(
            pair.path_pixels if pair.path_pixels is not None else [pair.init_pixels],
            pair.oracle_actions,
        )
        z = torch.stack(
            [_encode_frame(model, fr, transform, device) for fr in frames], dim=0
        )
        z_goal = _encode_frame(model, pair.goal_pixels, transform, device)
        d_start = float(torch.linalg.vector_norm(z[0] - z_goal))
        z_true_list.append(z.numpy())
        z_star_list.append(z_goal.numpy())
        d_start_list.append(d_start)
        if pair.path_state is not None:
            st = np.asarray(pair.path_state, dtype=np.float32)
            if len(st) < len(frames):
                pad = np.repeat(st[-1:], len(frames) - len(st), axis=0)
                st = np.concatenate([st, pad], axis=0)
            path_state_list.append(st[: len(frames)])
        else:
            path_state_list.append(
                np.zeros((len(frames), 7), dtype=np.float32)
            )
        for m in m_values:
            z_hat = imagine_closed_loop(model, z, acts, m, device=device)
            hats[m].append(z_hat.numpy())
            d = float(torch.linalg.vector_norm(z_hat[-1] - z_goal))
            d_end[m].append(d)
            toward[m].append(bool(d < d_start))
        print(f"pair {i + 1}/{len(pairs)} d_start={d_start:.3f} d_end_m25={d_end[m_values[-1]][-1]:.3f}")

    z_true = np.stack(z_true_list, axis=0).astype(np.float32)
    n_m = len(m_values)
    z_hat = np.stack([np.stack(hats[m], axis=0) for m in m_values], axis=1).astype(np.float32)
    d_end_arr = np.stack([np.asarray(d_end[m], dtype=np.float32) for m in m_values], axis=1)
    toward_arr = np.stack([np.asarray(toward[m], dtype=np.bool_) for m in m_values], axis=1)
    d_start_arr = np.asarray(d_start_list, dtype=np.float32)
    path_state = np.stack(path_state_list, axis=0).astype(np.float32)

    np.savez_compressed(
        out_dir / "ca0.npz",
        z_true=z_true,
        z_star=np.stack(z_star_list, axis=0).astype(np.float32),
        z_hat=z_hat,
        d_end=d_end_arr,
        d_start=d_start_arr,
        toward=toward_arr,
        m_values=np.asarray(m_values, dtype=np.int32),
        path_state=path_state,
        episode_id=np.arange(len(pairs), dtype=np.int32),
    )

    by_m = {}
    for j, m in enumerate(m_values):
        by_m[m] = {
            "m": m,
            "mean_d_end": float(d_end_arr[:, j].mean()),
            "mean_d_start": float(d_start_arr.mean()),
            "frac_toward": float(toward_arr[:, j].mean()),
            "n": int(len(pairs)),
        }
    fork = decide_fork(by_m)
    summary = {
        "oracle_bank": str(args.oracle_bank),
        "action_pack": "tile_block",
        "n_pairs": len(pairs),
        "m_values": m_values,
        "history": HISTORY,
        "by_m": by_m,
        "fork": fork["fork"],
        "fork_reason": fork["reason"],
        "thresholds": {
            "accum_toward_at_m5": ACCUM_TOWARD_AT_M5,
            "accum_d_end_at_m5": ACCUM_D_END_AT_M5,
            "m1_toward_guard": M1_TOWARD_GUARD,
            "m1_d_end_guard": M1_D_END_GUARD,
        },
        "bank_meta": bank_meta,
        "note": (
            "CA0 re-encodes true future observations (diagnostic, not a planner). "
            "m=25 should match C0 open-loop (~8.23 / 2% toward). "
            "Fork is this JSON, not a figure."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"fork": fork, "by_m": by_m}, indent=2))
    print(f"wrote {out_dir / 'ca0.npz'} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
