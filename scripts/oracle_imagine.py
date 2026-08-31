#!/usr/bin/env python3
"""C0.3-redo: imagine oracle actions vs encoded goal.

  python scripts/oracle_imagine.py \\
      --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0/ --device cuda

  python scripts/oracle_imagine.py \\
      --rollouts eval_results/pusht/c0_oracle_goal/oracle_rollouts.npz --device cuda
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
from eval_logging.extractors import pusht_pose_errors  # noqa: E402
from eval_logging.oracle_bank import load_oracle_bank  # noqa: E402
from eval_logging.runner import load_oracle_rollouts  # noqa: E402
from eval_setup import load_lewm_checkpoint  # noqa: E402
from phase_b import (  # noqa: E402
    HISTORY,
    encode_frames,
    imagine_path,
    img_transform,
)
from phi_data import frame_to_tensor  # noqa: E402


@torch.no_grad()
def _encode_frame(model, frame, transform, device) -> torch.Tensor:
    pix = frame_to_tensor(frame, transform).unsqueeze(0).to(device)
    return encode_frames(model, pix)[0].cpu()


def _rows_from_rollouts(model, transform, device, data) -> list[dict]:
    actions = data["action"]
    lengths = data["length"].astype(int)
    init_pix = data["init_pixels"]
    goal_pix = data["goal_pixels"]
    init_state = data["init_state"]
    goal_state = data["goal_state"]
    success = data["success"]
    rollout_pix = data.get("rollout_pixels")
    rows = []
    n = len(lengths)
    for i in range(n):
        t = int(lengths[i])
        acts = actions[i, :t]
        frames = [np.asarray(init_pix[i])]
        if rollout_pix is not None and n:
            rp = rollout_pix[i]
            if rp is not None and len(rp):
                frames.extend([np.asarray(fr) for fr in rp[:t]])
        rows.append(
            _imagine_one(
                model,
                transform,
                device,
                frames=frames,
                acts=acts,
                goal_pix=goal_pix[i],
                init_state=init_state[i],
                goal_state=goal_state[i],
                episode=i,
                env_success=bool(success[i]) if i < len(success) else False,
                rollout_state=(
                    data["rollout_state"][i] if "rollout_state" in data else None
                ),
            )
        )
    return rows


def _rows_from_bank(model, transform, device, pairs) -> list[dict]:
    rows = []
    for i, p in enumerate(pairs):
        acts = np.asarray(p.oracle_actions, dtype=np.float32)
        if p.path_pixels is not None:
            frames = [np.asarray(fr) for fr in p.path_pixels]
        else:
            frames = [np.asarray(p.init_pixels)]
        rows.append(
            _imagine_one(
                model,
                transform,
                device,
                frames=frames,
                acts=acts,
                goal_pix=p.goal_pixels,
                init_state=p.init_state,
                goal_state=p.goal_state,
                episode=i,
                env_success=True,
                rollout_state=p.path_state,
            )
        )
    return rows


def _imagine_one(
    model,
    transform,
    device,
    *,
    frames,
    acts,
    goal_pix,
    init_state,
    goal_state,
    episode: int,
    env_success: bool,
    rollout_state,
) -> dict:
    L = len(frames)
    if L < HISTORY + 1:
        while len(frames) < HISTORY + 1:
            frames.append(frames[0])
        L = len(frames)
    z = torch.stack(
        [_encode_frame(model, fr, transform, device) for fr in frames], dim=0
    )
    z_goal = _encode_frame(model, goal_pix, transform, device)
    acts = np.asarray(acts, dtype=np.float32)
    if acts.ndim == 1:
        acts = acts.reshape(1, -1)
    if len(acts) < L:
        pad = np.zeros(
            (L - len(acts), acts.shape[-1] if acts.size else 2), dtype=np.float32
        )
        acts = np.concatenate([acts, pad], axis=0) if acts.size else pad
    elif len(acts) > L:
        acts = acts[:L]
    z_hat = imagine_path(model, z, acts, device=device)
    d_hat = float(torch.linalg.vector_norm(z_hat[-1] - z_goal))
    d_true = float(torch.linalg.vector_norm(z[-1] - z_goal))
    d_start = float(torch.linalg.vector_norm(z[0] - z_goal))
    pos0, _ = pusht_pose_errors(goal_state, init_state)
    pos_end = float("nan")
    if rollout_state is not None and len(np.asarray(rollout_state)):
        st = np.asarray(rollout_state)
        pos_end, _ = pusht_pose_errors(goal_state, st[-1].reshape(-1))
    return {
        "episode": episode,
        "env_success": bool(env_success),
        "n_actions": int(len(acts)),
        "d_start_z": d_start,
        "d_true_end_z": d_true,
        "d_hat_end_z": d_hat,
        "imag_vs_true": d_hat / max(d_true, 1e-6),
        "imag_moved_toward_goal": d_hat < d_start,
        "pos_start": float(pos0),
        "pos_end": float(pos_end) if pos_end == pos_end else None,
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=Path, default=None)
    p.add_argument("--oracle-bank", type=Path, default=None)
    p.add_argument("--env", default="pusht")
    p.add_argument("--device", default="cuda")
    p.add_argument("--pack", default="tile_block")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    if bool(args.rollouts) == bool(args.oracle_bank):
        raise SystemExit("pass exactly one of --oracle-bank or --rollouts")
    if args.pack != "tile_block":
        raise SystemExit("only --pack tile_block is implemented")

    spec = ENV_REGISTRY[args.env]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_lewm_checkpoint(spec.ckpt_dir)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    transform = img_transform(spec.img_size)

    source = "oracle_bank" if args.oracle_bank else "rollouts"
    if args.oracle_bank:
        pairs, meta = load_oracle_bank(args.oracle_bank)
        rows = _rows_from_bank(model, transform, device, pairs)
        out_dir = Path(args.out_dir or args.oracle_bank)
        extra_meta = {"oracle_source": meta.get("oracle_source"), "pair_band": meta.get("pair_band")}
    else:
        data = load_oracle_rollouts(args.rollouts)
        rows = _rows_from_rollouts(model, transform, device, data)
        out_dir = Path(args.out_dir or args.rollouts.parent)
        extra_meta = {}

    moved = [r["imag_moved_toward_goal"] for r in rows]
    env_ok = [r for r in rows if r["env_success"]]
    fidelity_fail = [
        r
        for r in env_ok
        if r["imag_vs_true"] > 2.0 or not r["imag_moved_toward_goal"]
    ]
    summary = {
        "source": source,
        "rollouts": str(args.rollouts) if args.rollouts else None,
        "oracle_bank": str(args.oracle_bank) if args.oracle_bank else None,
        "action_pack": "tile_block",
        "n": len(rows),
        "n_env_success": int(sum(1 for r in rows if r["env_success"])),
        "mean_d_hat_end": float(np.mean([r["d_hat_end_z"] for r in rows])) if rows else None,
        "mean_d_true_end": float(np.mean([r["d_true_end_z"] for r in rows])) if rows else None,
        "frac_imag_moved_toward_goal": float(np.mean(moved)) if moved else None,
        "n_model_fidelity_fail_on_env_success": len(fidelity_fail),
        "note": (
            "Toward-goal = ‖ẑ_end−z*‖ < ‖z_start−z*‖. Outcome B if this "
            "fraction is low on actions that succeed in env (oracle-replay)."
        ),
        **extra_meta,
        "episodes": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "oracle_imagine.json"
    path.write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "n",
                    "n_env_success",
                    "mean_d_hat_end",
                    "mean_d_true_end",
                    "frac_imag_moved_toward_goal",
                    "n_model_fidelity_fail_on_env_success",
                )
            },
            indent=2,
        )
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
