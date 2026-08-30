#!/usr/bin/env python3
"""Thorough pre-retrain confirmation of the Part A INFIDELITY reading.

Frozen cuts live in thresholds.yaml ``infidelity_confirm`` (same frac as encoder_floor).
Do not retune after seeing numbers. Does not start Part B.

  python scripts/infidelity_confirm.py \\
      --ca0 eval_results/pusht/ca0_closed_loop/seed0 \\
      --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0 \\
      --out eval_results/pusht/infidelity_confirm \\
      --seeds 1 2 --device cuda
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

from encoder_floor import (  # noqa: E402
    _pad_frames_actions,
    bracket_frac,
    decide_guard,
    dump_metrics,
    onestep_errors,
)
from phase_b import HISTORY, contact_events, imagine_closed_loop, shuffle_future_actions  # noqa: E402
from viz import load_ca0, load_thresholds  # noqa: E402


def _spearman(x, y) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(len(x), len(y))
    if n < 8:
        return None
    x, y = x[:n], y[:n]
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    if den <= 1e-12:
        return None
    return float((rx * ry).sum() / den)


def _median_frac(err, adj, floor: float) -> float | None:
    if len(err) < 5:
        return None
    return bracket_frac(float(np.median(err)), floor, float(np.median(adj)))


def _angles_deg(true_d, pred_d) -> np.ndarray:
    out = []
    for a, b in zip(true_d, pred_d):
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            continue
        c = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        out.append(float(np.degrees(np.arccos(c))))
    return np.asarray(out, dtype=np.float64)


def frac_bundle(z_true: np.ndarray, z_hat: np.ndarray, *, floor: float = 0.0) -> dict:
    """Teacher-forced one-step bracket on aligned (N, L, D) tensors."""
    z_true = np.asarray(z_true, dtype=np.float64)
    z_hat = np.asarray(z_hat, dtype=np.float64)
    err = onestep_errors(z_hat, z_true)
    adj = np.linalg.norm(z_true[:, HISTORY:] - z_true[:, HISTORY - 1 : -1], axis=-1).reshape(-1)
    n = min(len(err), len(adj))
    err, adj = err[:n], adj[:n]
    frac = _median_frac(err, adj, floor)
    return {
        "n_steps": int(n),
        "n_pairs": int(z_true.shape[0]),
        "onestep_median": float(np.median(err)),
        "onestep_mean": float(np.mean(err)),
        "adjacent_median": float(np.median(adj)),
        "frac": frac,
        "guard": decide_guard(frac, calib_ok=True, th=load_thresholds().get("encoder_floor") or {}),
        "err": err,
        "adj": adj,
    }


def geometry_from_ca0(ca0: dict, *, floor: float = 0.0, tercile_edges: tuple[float, float] | None = None) -> dict:
    """Mechanism: identity vs wrong-map, pose, strata, terciles.

    ``tercile_edges`` = (q33, q67) frozen from the seed-0 live-bank adjacent-Δz
    pre distribution. If omitted, edges are fit on *this* bank (A-confirm only).
    """
    z_true = np.asarray(ca0["z_true"], dtype=np.float64)
    m_values = [int(x) for x in ca0["m_values"]]
    i1 = m_values.index(1) if 1 in m_values else 0
    hat = np.asarray(ca0["z_hat"][:, i1], dtype=np.float64)
    bundle = frac_bundle(z_true, hat, floor=floor)
    err, adj = bundle["err"], bundle["adj"]
    pred_d = hat[:, HISTORY:] - z_true[:, HISTORY - 1 : -1]
    true_d = z_true[:, HISTORY:] - z_true[:, HISTORY - 1 : -1]
    pred_move = np.linalg.norm(pred_d, axis=-1).reshape(-1)
    n = min(len(err), len(pred_move))
    pred_move = pred_move[:n]
    angles = _angles_deg(true_d.reshape(-1, true_d.shape[-1]), pred_d.reshape(-1, pred_d.shape[-1]))
    move_ratio = float(np.median(pred_move) / max(float(np.median(adj)), 1e-8))
    qs = (
        np.asarray(tercile_edges, dtype=np.float64)
        if tercile_edges is not None
        else np.quantile(adj, [1.0 / 3.0, 2.0 / 3.0])
    )
    terciles = {}
    for name, mask in (
        ("small", adj <= qs[0]),
        ("mid", (adj > qs[0]) & (adj <= qs[1])),
        ("large", adj > qs[1]),
    ):
        terciles[name] = {
            "n": int(mask.sum()),
            "adjacent_median": float(np.median(adj[mask])) if mask.any() else None,
            "onestep_median": float(np.median(err[mask])) if mask.any() else None,
            "frac": _median_frac(err[mask], adj[mask], floor) if mask.sum() >= 5 else None,
        }

    pose = {}
    strata = {}
    if "path_state" in ca0:
        st = np.asarray(ca0["path_state"], dtype=np.float64)
        L = min(st.shape[1], z_true.shape[1])
        st = st[:, :L]
        agent_d = np.linalg.norm(st[:, HISTORY:, :2] - st[:, HISTORY - 1 : -1, :2], axis=-1).reshape(-1)
        block_d = np.linalg.norm(st[:, HISTORY:, 2:4] - st[:, HISTORY - 1 : -1, 2:4], axis=-1).reshape(-1)
        n2 = min(len(adj), len(block_d), len(agent_d))
        pose = {
            "agent_xy_step_median": float(np.median(agent_d[:n2])),
            "block_xy_step_median": float(np.median(block_d[:n2])),
            "spearman_adj_vs_block": _spearman(adj[:n2], block_d[:n2]),
            "spearman_adj_vs_agent": _spearman(adj[:n2], agent_d[:n2]),
            "spearman_adj_vs_pose": _spearman(adj[:n2], np.sqrt(block_d[:n2] ** 2 + agent_d[:n2] ** 2)),
        }
        ev = contact_events(st, env="pusht")
        tgt = {k: np.asarray(v)[:, HISTORY:].reshape(-1)[:n] for k, v in ev.items()}
        for name in ("contact", "wall", "free"):
            if name == "free":
                mask = ~tgt.get("any", np.zeros(n, dtype=bool))
            else:
                mask = tgt.get(name, np.zeros(n, dtype=bool))
            strata[name] = {
                "n": int(mask.sum()),
                "frac": _median_frac(err[mask], adj[mask], floor) if mask.sum() >= 5 else None,
                "onestep_median": float(np.median(err[mask])) if mask.any() else None,
                "adjacent_median": float(np.median(adj[mask])) if mask.any() else None,
            }
        moving = block_d[:n2] > 1.0
        parked = ~moving
        pose["frac_on_block_moving_steps"] = (
            _median_frac(err[:n2][moving], adj[:n2][moving], floor) if moving.sum() >= 5 else None
        )
        pose["frac_on_block_parked_steps"] = (
            _median_frac(err[:n2][parked], adj[:n2][parked], floor) if parked.sum() >= 5 else None
        )
        pose["n_block_moving_steps"] = int(moving.sum())
        pose["n_block_parked_steps"] = int(parked.sum())

    out = {
        "frac": bundle["frac"],
        "guard": bundle["guard"],
        "onestep_median": bundle["onestep_median"],
        "adjacent_median": bundle["adjacent_median"],
        "predicted_move_median": float(np.median(pred_move)),
        "predicted_move_over_adjacent": move_ratio,
        "mean_angle_deg": float(np.mean(angles)) if angles.size else None,
        "median_angle_deg": float(np.median(angles)) if angles.size else None,
        "n_angles": int(angles.size),
        "n_steps": bundle["n_steps"],
        "tercile_edges": [float(qs[0]), float(qs[1])],
        "terciles": terciles,
        "pose": pose,
        "strata": strata,
    }
    return out


def decide_confirm(payload: dict, th: dict) -> dict:
    """Classify *where* infidelity lives. Does not retune encoder_floor cuts."""
    accum = float(th.get("frac_accum_at_or_below", 0.25))
    inf = float(th.get("frac_infidelity_at_or_above", 0.5))
    spear_cut = float(th.get("pose_z_spearman_min", 0.30))
    ident_cut = float(th.get("predicted_move_over_adjacent_identity_below", 0.25))
    deaf_cut = float(th.get("shuffle_gap_over_adjacent_deaf_below", 0.10))

    def _f(arm):
        if not arm:
            return None
        return arm.get("frac")

    live = _f(payload.get("geometry"))
    diverse = _f(payload.get("diverse"))
    seed_fracs = []
    for s in payload.get("seeds") or []:
        seed_fracs.append(_f(s))
    pose_s = ((payload.get("geometry") or {}).get("pose") or {}).get("spearman_adj_vs_pose")
    move_r = (payload.get("geometry") or {}).get("predicted_move_over_adjacent")
    shuf = payload.get("shuffle") or {}
    shuf_gap = shuf.get("gap_over_adjacent")

    flags = []
    if pose_s is not None and pose_s < spear_cut:
        flags.append("ENCODER_JITTER")
    if move_r is not None and move_r < ident_cut:
        flags.append("FROZEN_IDENTITY")
    if shuf_gap is not None and shuf_gap < deaf_cut:
        flags.append("ACTION_DEAF_LIVEBANK")
    if any(f is not None and f <= accum for f in seed_fracs):
        flags.append("SEED_FLUKE")
    if diverse is not None and live is not None and diverse <= accum and live >= inf:
        flags.append("BANK_SPECIFIC")

    seeds_all_inf = bool(seed_fracs) and all(f is not None and f >= inf for f in seed_fracs)
    diverse_inf = diverse is not None and diverse >= inf
    live_inf = live is not None and live >= inf

    if "SEED_FLUKE" in flags:
        overall = "SEED_FLUKE"
        retrain = False
        reason = "A later seed has frac ≤ 0.25. Seed-0 INFIDELITY is not confirmed. Do not start Part B."
    elif "ENCODER_JITTER" in flags:
        overall = "ENCODER_JITTER"
        retrain = False
        reason = "Adjacent Δz does not track sim pose. Investigate the encoder before a P retrain."
    elif "BANK_SPECIFIC" in flags:
        overall = "BANK_SPECIFIC"
        retrain = True
        reason = (
            "Live-bank (GoalPush windows) is infidelity; random-action bank is not. "
            "Part B data must include on-policy windows; do not treat P as universally broken."
        )
    elif live_inf and (not seed_fracs or seeds_all_inf) and (diverse is None or diverse_inf):
        overall = "CONFIRMED_INFIDELITY"
        retrain = True
        reason = "frac ≥ 0.5 on live-bank and every extra bank that ran. Part B is still gated on, not started."
    else:
        overall = "MIXED"
        retrain = live_inf or (diverse is not None and diverse >= inf) or any(
            f is not None and f > accum for f in seed_fracs
        )
        reason = "Banks disagree in degree but none flipped to ACCUMULATION. Record and proceed only if live-bank still INFIDELITY."

    return {
        "overall": overall,
        "gate_part_b": bool(retrain) and overall not in ("SEED_FLUKE", "ENCODER_JITTER"),
        "flags": flags,
        "reason": reason,
        "live_frac": live,
        "diverse_frac": diverse,
        "seed_fracs": seed_fracs,
        "pose_spearman": pose_s,
        "predicted_move_over_adjacent": move_r,
        "shuffle_gap_over_adjacent": shuf_gap,
        "cuts": {
            "frac_accum_at_or_below": accum,
            "frac_infidelity_at_or_above": inf,
            "pose_z_spearman_min": spear_cut,
        },
    }


def _load_model(ckpt: str, device_s: str):
    import torch

    from eval_setup import load_lewm_checkpoint
    from phase_b import img_transform

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(device_s if torch.cuda.is_available() else "cpu")
    model = load_lewm_checkpoint(ckpt)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, device, img_transform(224)


def _m1_from_z(model, z: np.ndarray, acts: np.ndarray, device):
    import torch

    z_t = torch.from_numpy(np.asarray(z, dtype=np.float32))
    L = min(int(z_t.shape[0]), len(acts))
    return imagine_closed_loop(model, z_t[:L], acts[:L], m=1, device=device).numpy()


def _pairs_m1(model, pairs, z_true, device, *, action_mode: str, rng):
    hats = []
    n = min(len(pairs), len(z_true))
    for i in range(n):
        pair = pairs[i]
        _, acts, _ = _pad_frames_actions(
            pair.path_pixels if pair.path_pixels is not None else [pair.init_pixels],
            pair.oracle_actions,
        )
        acts = np.asarray(acts, dtype=np.float32)
        if action_mode == "shuffle":
            acts = shuffle_future_actions(acts, HISTORY, rng)
        elif action_mode == "zero":
            acts = np.zeros_like(acts)
        hats.append(_m1_from_z(model, z_true[i], acts, device))
        if (i + 1) % 10 == 0:
            print(f"  {action_mode} pair {i + 1}/{n}")
    return np.stack(hats, axis=0)


def _encode_frames(model, frames, transform, device):
    import torch

    from phase_b import encode_frames
    from phi_data import frame_to_tensor

    zs = []
    for fr in frames:
        pix = frame_to_tensor(fr, transform).unsqueeze(0).to(device)
        zs.append(encode_frames(model, pix)[0].detach().float().cpu())
    return torch.stack(zs, dim=0).numpy()


def _run_diverse(model, device, transform, args) -> dict:
    from eval_live import ENV_REGISTRY
    from eval_logging.pairs import collect_trajectory_bank
    from latent_dump import _episode_actions, _state_block, sample_segments
    import stable_worldmodel as swm

    spec = ENV_REGISTRY["pusht"]
    world = swm.World(
        env_name=spec.env_name,
        num_envs=1,
        max_episode_steps=spec.max_episode_steps,
        image_shape=(spec.img_size, spec.img_size),
        **spec.world_kwargs,
    )
    print(f"collecting random bank steps={args.diverse_steps}")
    try:
        bank = collect_trajectory_bank(
            world,
            num_steps=args.diverse_steps,
            seed=args.diverse_seed,
            env_name=spec.env_name,
            min_episode_len=args.segment_len,
            collector="random",
            num_episodes=None,
        )
    finally:
        world.close()
    segs = sample_segments(
        bank,
        n_segments=args.n_segments,
        segment_len=args.segment_len,
        seed=args.diverse_seed,
        min_length=args.segment_len,
    )
    z_list, hat_list, st_list = [], [], []
    rng = np.random.default_rng(args.diverse_seed + 3)
    shuf_hats = []
    for j, (ep_i, start) in enumerate(segs):
        ep = bank.episodes[ep_i]
        L = args.segment_len
        frames = [np.asarray(ep.pixels[t]) for t in range(start, start + L)]
        z = _encode_frames(model, frames, transform, device)
        acts = _episode_actions(ep, start, L)
        hat = _m1_from_z(model, z, acts, device)
        shuf = shuffle_future_actions(acts, HISTORY, rng)
        hat_s = _m1_from_z(model, z, shuf, device)
        z_list.append(z)
        hat_list.append(hat)
        shuf_hats.append(hat_s)
        st_list.append(_state_block(ep, start, L))
        if (j + 1) % 10 == 0:
            print(f"  diverse segment {j + 1}/{len(segs)}")
    z_true = np.stack(z_list, axis=0)
    z_hat = np.stack(hat_list, axis=0)
    z_shuf = np.stack(shuf_hats, axis=0)
    true_b = frac_bundle(z_true, z_hat)
    shuf_b = frac_bundle(z_true, z_shuf)
    adj = true_b["adjacent_median"]
    gap = float(shuf_b["onestep_median"] - true_b["onestep_median"])
    return {
        "n_segments": int(len(segs)),
        "n_bank_episodes": int(len(bank.episodes)),
        "collector": "random",
        "frac": true_b["frac"],
        "guard": true_b["guard"],
        "onestep_median": true_b["onestep_median"],
        "adjacent_median": true_b["adjacent_median"],
        "n_steps": true_b["n_steps"],
        "shuffle_onestep_median": shuf_b["onestep_median"],
        "shuffle_frac": shuf_b["frac"],
        "gap_over_adjacent": float(gap / max(adj, 1e-8)),
        "z_true": z_true,
        "z_hat": z_hat,
        "path_state": np.stack(st_list, axis=0),
    }


def _run_seed(model, device, transform, seed: int, args) -> dict:
    from eval_logging.oracle_bank import load_oracle_bank
    from oracle_bank import main as collect_oracle_bank

    bank_dir = Path(args.oracle_bank_root) / f"seed{seed}"
    if not (bank_dir / "pairs.npz").exists():
        collect_oracle_bank(
            [
                "--env",
                "pusht",
                "--controller",
                "goalpush",
                "--collect",
                str(args.collect_episodes),
                "--window",
                "25",
                "--band",
                "short_horizon",
                "--pack",
                "tile_block",
                "--seed",
                str(seed),
                "--num-pairs",
                "50",
                "--out",
                str(bank_dir),
            ]
        )
    pairs, meta = load_oracle_bank(bank_dir)
    z_true_list, z_star_list, hat_list, state_list = [], [], [], []
    d_end = []
    print(f"seed {seed}: encoding {len(pairs)} pairs")
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
        if pair.path_state is not None:
            st = np.asarray(pair.path_state, dtype=np.float32)
            if len(st) < len(frames):
                st = np.concatenate([st, np.repeat(st[-1:], len(frames) - len(st), axis=0)], axis=0)
            state_list.append(st[: len(frames)])
        else:
            state_list.append(np.zeros((len(frames), 7), dtype=np.float32))
        if (i + 1) % 10 == 0:
            print(f"  seed {seed} pair {i + 1}/{len(pairs)}")
    z_true = np.stack(z_true_list, axis=0).astype(np.float32)
    z_hat = np.stack(hat_list, axis=0).astype(np.float32)
    ca0 = {
        "z_true": z_true,
        "z_star": np.stack(z_star_list, axis=0).astype(np.float32),
        "z_hat": z_hat[:, None],
        "m_values": np.array([1], dtype=np.int32),
        "d_end": np.asarray(d_end, dtype=np.float32)[:, None],
        "d_start": np.linalg.norm(z_true[:, 0] - np.stack(z_star_list), axis=-1).astype(np.float32),
        "toward": (np.asarray(d_end) < np.linalg.norm(z_true[:, 0] - np.stack(z_star_list), axis=-1))[:, None],
        "path_state": np.stack(state_list, axis=0).astype(np.float32),
        "episode_id": np.arange(len(pairs), dtype=np.int32),
    }
    out_ca0 = Path(args.out) / f"ca0_seed{seed}"
    out_ca0.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_ca0 / "ca0.npz", **ca0)
    dump = dump_metrics(ca0, {"by_m": {"1": {"mean_d_end": float(np.mean(d_end))}}})
    frac = bracket_frac(dump["onestep_median"], 0.0, dump["adjacent_median"])
    summary = {
        "seed": seed,
        "oracle_bank": str(bank_dir),
        "n_pairs": len(pairs),
        "bank_meta": meta,
        "fork_mean_d_end_m1": dump["fork_mean_d_end_m1"],
        "onestep_median": dump["onestep_median"],
        "adjacent_median": dump["adjacent_median"],
        "frac": frac,
        "guard": decide_guard(frac, calib_ok=True, th=load_thresholds().get("encoder_floor") or {}),
        "geometry": geometry_from_ca0(ca0),
    }
    (out_ca0 / "summary.json").write_text(json.dumps({k: summary[k] for k in summary if k != "bank_meta"}, indent=2, default=str))
    return summary


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ca0", type=Path, required=True)
    p.add_argument("--oracle-bank", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ckpt", default="hf_pusht")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    p.add_argument("--oracle-bank-root", type=Path, default=None)
    p.add_argument("--collect-episodes", type=int, default=60)
    p.add_argument("--diverse-steps", type=int, default=8000)
    p.add_argument("--diverse-seed", type=int, default=0)
    p.add_argument("--n-segments", type=int, default=80)
    p.add_argument("--segment-len", type=int, default=26)
    p.add_argument("--geometry-only", action="store_true")
    p.add_argument("--skip-diverse", action="store_true")
    p.add_argument("--skip-seeds", action="store_true")
    p.add_argument("--skip-shuffle", action="store_true")
    args = p.parse_args(argv)
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.oracle_bank_root is None:
        args.oracle_bank_root = Path(args.oracle_bank).resolve().parent

    th = load_thresholds().get("infidelity_confirm") or {}
    ca0, _ = load_ca0(args.ca0)
    geometry = geometry_from_ca0(ca0)
    payload = {"geometry": geometry, "shuffle": None, "zero": None, "diverse": None, "seeds": []}
    (args.out / "geometry.json").write_text(json.dumps(geometry, indent=2, default=str))
    print("geometry", json.dumps({k: geometry[k] for k in ("frac", "guard", "predicted_move_over_adjacent", "mean_angle_deg", "pose", "strata")}, indent=2, default=str))

    if args.geometry_only:
        decision = decide_confirm(payload, th)
        (args.out / "summary.json").write_text(json.dumps({"decision": decision, **_jsonable(payload)}, indent=2, default=str))
        print(json.dumps(decision, indent=2))
        return

    from eval_logging.oracle_bank import load_oracle_bank

    model, device, transform = _load_model(args.ckpt, args.device)
    pairs, _ = load_oracle_bank(args.oracle_bank)
    z_true = np.asarray(ca0["z_true"])

    if not args.skip_shuffle:
        rng = np.random.default_rng(0)
        print("live-bank shuffle + zero m=1")
        true_hat = _pairs_m1(model, pairs, z_true, device, action_mode="true", rng=rng)
        shuf_hat = _pairs_m1(model, pairs, z_true, device, action_mode="shuffle", rng=rng)
        zero_hat = _pairs_m1(model, pairs, z_true, device, action_mode="zero", rng=rng)
        tb, sb, zb = (frac_bundle(z_true, h) for h in (true_hat, shuf_hat, zero_hat))
        adj = tb["adjacent_median"]
        payload["shuffle"] = {
            "frac": sb["frac"],
            "onestep_median": sb["onestep_median"],
            "true_onestep_median": tb["onestep_median"],
            "gap_over_adjacent": float((sb["onestep_median"] - tb["onestep_median"]) / max(adj, 1e-8)),
            "n_steps": sb["n_steps"],
        }
        payload["zero"] = {
            "frac": zb["frac"],
            "onestep_median": zb["onestep_median"],
            "true_onestep_median": tb["onestep_median"],
            "gap_over_adjacent": float((zb["onestep_median"] - tb["onestep_median"]) / max(adj, 1e-8)),
            "n_steps": zb["n_steps"],
        }
        print("shuffle", json.dumps(payload["shuffle"], indent=2))
        print("zero", json.dumps(payload["zero"], indent=2))

    if not args.skip_diverse:
        print("diverse random-action teacher-forced m=1")
        diverse = _run_diverse(model, device, transform, args)
        geom_d = geometry_from_ca0(
            {
                "z_true": diverse["z_true"],
                "z_hat": diverse["z_hat"][:, None],
                "m_values": np.array([1], dtype=np.int32),
                "path_state": diverse["path_state"],
            }
        )
        payload["diverse"] = {k: diverse[k] for k in diverse if k not in ("z_true", "z_hat", "path_state")}
        payload["diverse"]["geometry"] = geom_d
        print("diverse", json.dumps(payload["diverse"], indent=2, default=str))

    if not args.skip_seeds:
        for seed in args.seeds:
            print(f"===== seed {seed} oracle bank + m=1 =====")
            payload["seeds"].append(_run_seed(model, device, transform, int(seed), args))

    decision = decide_confirm(payload, th)
    out = {"decision": decision, **_jsonable(payload)}
    (args.out / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(decision, indent=2))
    print(f"wrote {args.out / 'summary.json'}")


if __name__ == "__main__":
    main()
