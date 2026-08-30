#!/usr/bin/env python3
"""Part A: calibrate the encoder ruler, then bracket one-step error (doc 16).

Two different numbers — do not mix them:
  * fork metric    = CA0 m=1 mean ‖ẑ_end − z*‖  (was 1.43; distance to goal)
  * bracket metric = one-step ‖P(true history, a) − z_{t+1}‖  (this is `frac`)

  python scripts/encoder_floor.py \\
      --ca0 eval_results/pusht/ca0_closed_loop/seed0 \\
      --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0 \\
      --out eval_results/pusht/encoder_floor/seed0 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

from phase_b import HISTORY  # noqa: E402
from viz import load_ca0, load_thresholds  # noqa: E402


def bracket_frac(error: float, floor: float, null: float) -> float | None:
    """(error − perfect) / (identity − perfect). None if the ruler has no width."""
    denom = float(null) - float(floor)
    if denom <= 1e-8:
        return None
    return float((float(error) - float(floor)) / denom)


def decide_guard(frac: float | None, *, calib_ok: bool, th: dict) -> str:
    """Pre-registered cuts. Do not retune after seeing frac."""
    if not calib_ok or frac is None:
        return "CALIB_FAIL"
    accum = float(th.get("frac_accum_at_or_below", 0.25))
    inf = float(th.get("frac_infidelity_at_or_above", 0.5))
    if frac <= accum:
        return "ACCUMULATION"
    if frac >= inf:
        return "INFIDELITY"
    return "PARTIAL"


def _rel_agree(a: float, b: float) -> float:
    scale = max(abs(float(a)), abs(float(b)), 1e-8)
    return abs(float(a) - float(b)) / scale


def onestep_errors(z_hat_m1: np.ndarray, z_true: np.ndarray, history: int = HISTORY) -> np.ndarray:
    """‖ẑ_t − z_t‖ for t ≥ history (teacher-forced one-step at m=1)."""
    z_hat_m1 = np.asarray(z_hat_m1, dtype=np.float64)
    z_true = np.asarray(z_true, dtype=np.float64)
    pred = slice(history, None)
    return np.linalg.norm(z_hat_m1[:, pred] - z_true[:, pred], axis=-1).reshape(-1)


def dump_metrics(ca0: dict, summary: dict | None = None) -> dict:
    """Numbers that live in ca0.npz (no GPU). Fork metric ≠ one-step metric."""
    z_true = np.asarray(ca0["z_true"], dtype=np.float64)
    z_star = np.asarray(ca0["z_star"], dtype=np.float64)
    z_hat = np.asarray(ca0["z_hat"], dtype=np.float64)
    d_end = np.asarray(ca0["d_end"], dtype=np.float64)
    m_values = [int(x) for x in ca0["m_values"]]
    i1 = m_values.index(1) if 1 in m_values else 0
    hat1 = z_hat[:, i1]
    onestep = onestep_errors(hat1, z_true)
    adj = np.linalg.norm(z_true[:, 1:] - z_true[:, :-1], axis=-1).reshape(-1)
    last_vs_star = np.linalg.norm(z_true[:, -1] - z_star, axis=-1)
    rng = np.random.default_rng(0)
    flat = z_true.reshape(-1, z_true.shape[-1])
    n = min(400, len(flat))
    ii = rng.choice(len(flat), size=n, replace=False)
    jj = rng.choice(len(flat), size=n, replace=False)
    spread = np.linalg.norm(flat[ii] - flat[jj], axis=-1)
    reported = None
    if summary and summary.get("by_m"):
        raw = summary["by_m"]
        entry = raw.get("1", raw.get(1, {})) or {}
        if "mean_d_end" in entry:
            reported = float(entry["mean_d_end"])
    adj_into = (
        np.linalg.norm(z_true[:, HISTORY:] - z_true[:, HISTORY - 1 : -1], axis=-1).reshape(-1)
        if z_true.shape[1] > HISTORY
        else adj
    )
    return {
        "n_pairs": int(len(z_true)),
        "history": int(HISTORY),
        "m1_index": int(i1),
        "fork_mean_d_end_m1": float(d_end[:, i1].mean()),
        "fork_reported_d_end_m1": reported,
        "mean_last_true_vs_zstar": float(last_vs_star.mean()),
        "median_last_true_vs_zstar": float(np.median(last_vs_star)),
        "onestep_mean": float(onestep.mean()),
        "onestep_median": float(np.median(onestep)),
        "onestep_p10": float(np.percentile(onestep, 10)),
        "onestep_p90": float(np.percentile(onestep, 90)),
        "adjacent_mean": float(adj.mean()),
        "adjacent_median": float(np.median(adj)),
        "random_pair_spread_median": float(np.median(spread)),
        "onestep": onestep,
        "adjacent_into_next": adj_into,
    }


def _pad_frames_actions(frames, acts):
    """Match scripts/closed_loop_imagine._pad_frames_actions (CA0 packing)."""
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


def _pcts(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"p10": None, "p50": None, "p90": None}
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _run_gpu(args, ca0: dict, dump: dict, th: dict) -> tuple[bool, list[str], dict, dict, float]:
    import torch

    from eval_logging.oracle_bank import load_oracle_bank
    from eval_setup import load_lewm_checkpoint
    from phase_b import encode_frames, imagine_closed_loop, img_transform
    from phi_data import frame_to_tensor

    calib_ok = True
    notes: list[str] = []
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_lewm_checkpoint(args.ckpt)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    transform = img_transform(224)

    if args.oracle_bank is None:
        raise SystemExit("GPU run needs --oracle-bank (live-bank, not kinematic phase_b_dump)")
    pairs, _ = load_oracle_bank(args.oracle_bank)

    frames: list[np.ndarray] = []
    for pair in pairs:
        pix = pair.path_pixels
        if pix is None:
            continue
        for fr in np.asarray(pix):
            frames.append(np.asarray(fr))
            if len(frames) >= args.n_det_frames:
                break
        if len(frames) >= args.n_det_frames:
            break
    if not frames:
        raise SystemExit("oracle bank has no path_pixels to encode")

    d_same = []
    for fr in frames:
        pix = frame_to_tensor(fr, transform).unsqueeze(0).to(device)
        z1 = encode_frames(model, pix)[0].detach().float().cpu().numpy()
        z2 = encode_frames(model, pix)[0].detach().float().cpu().numpy()
        d_same.append(float(np.linalg.norm(z1 - z2)))
    d_same = np.asarray(d_same, dtype=np.float64)
    pix0 = frame_to_tensor(frames[0], transform)
    batch = pix0.unsqueeze(0).repeat(8, 1, 1, 1).to(device)
    zb = encode_frames(model, batch).detach().float().cpu().numpy()
    batch_pair = float(np.linalg.norm(zb[0] - zb[-1]))
    floor = float(np.median(d_same))
    det_cut = float(th.get("determinism_median_max", 0.01))
    if floor > det_cut or batch_pair > det_cut:
        calib_ok = False
        notes.append(f"encoder not deterministic at eval (median={floor}, batch={batch_pair})")
    determinism = {
        "n_frames": int(len(d_same)),
        "same_state_reencode_median": float(np.median(d_same)),
        "same_state_reencode_mean": float(np.mean(d_same)),
        "same_state_reencode_max": float(np.max(d_same)),
        "batch_position_pair_l2": batch_pair,
        "eval_mode": True,
        "cudnn_deterministic": True,
        "tensor": "phase_b.encode_frames → encode({'pixels'})['emb'][:, 0]",
        "norm": "L2",
        "normalization": "none (raw encoder embedding, no SIGReg reshape at eval)",
    }

    # Dump z vs a fresh encode of the same frames (units / same space).
    pair0 = pairs[0]
    frames0, _, _ = _pad_frames_actions(
        pair0.path_pixels if pair0.path_pixels is not None else [pair0.init_pixels],
        pair0.oracle_actions,
    )
    n_chk = min(8, len(frames0), int(np.asarray(ca0["z_true"]).shape[1]))
    z_re = []
    for fr in frames0[:n_chk]:
        pix = frame_to_tensor(fr, transform).unsqueeze(0).to(device)
        z_re.append(encode_frames(model, pix)[0].detach().float().cpu().numpy())
    z_re = np.stack(z_re, axis=0)
    z_dump0 = np.asarray(ca0["z_true"][0, :n_chk], dtype=np.float64)
    dump_vs_enc = np.linalg.norm(z_re.astype(np.float64) - z_dump0, axis=-1)
    dump_vs_enc_rel = float(dump_vs_enc.mean() / max(np.linalg.norm(z_dump0, axis=-1).mean(), 1e-8))
    rel_cut = float(th.get("dump_vs_fresh_rel_max", 0.10))
    if dump_vs_enc_rel > rel_cut:
        calib_ok = False
        notes.append(f"re-encode vs dump z_true relative L2 {dump_vs_enc_rel:.3f} > {rel_cut}")
    determinism["dump_vs_reencode_mean_l2"] = float(dump_vs_enc.mean())
    determinism["dump_vs_reencode_rel"] = dump_vs_enc_rel

    # Fresh P, teacher-forced m=1, dump z_true + live-bank actions (same pad as CA0).
    hats = []
    n = min(len(pairs), int(np.asarray(ca0["z_true"]).shape[0]))
    z_true_all = np.asarray(ca0["z_true"], dtype=np.float32)
    for i in range(n):
        pair = pairs[i]
        _, acts, _ = _pad_frames_actions(
            pair.path_pixels if pair.path_pixels is not None else [pair.init_pixels],
            pair.oracle_actions,
        )
        z = torch.from_numpy(z_true_all[i])
        L = min(int(z.shape[0]), len(acts))
        z_hat = imagine_closed_loop(model, z[:L], acts[:L], m=1, device=device)
        hats.append(z_hat.numpy())
    fresh_hat = np.stack(hats, axis=0)
    i1 = dump["m1_index"]
    dump_hat = np.asarray(ca0["z_hat"][:, i1])
    nuse = min(len(fresh_hat), len(dump_hat))
    lf = min(fresh_hat.shape[1], dump_hat.shape[1], z_true_all.shape[1])
    z_true_f = np.asarray(ca0["z_true"], dtype=np.float64)
    e_fresh = onestep_errors(fresh_hat[:nuse, :lf], z_true_f[:nuse, :lf])
    e_dump = onestep_errors(dump_hat[:nuse, :lf], z_true_f[:nuse, :lf])
    hat_rel = _rel_agree(float(e_fresh.mean()), float(e_dump.mean()))
    if hat_rel > rel_cut:
        calib_ok = False
        notes.append(f"fresh P() one-step disagrees with ca0.npz m=1 ẑ (rel={hat_rel:.3f})")
    fresh = {
        "onestep_mean": float(e_fresh.mean()),
        "onestep_median": float(np.median(e_fresh)),
        "dump_onestep_mean_aligned": float(e_dump.mean()),
        "dump_vs_fresh_rel": hat_rel,
        "n_pairs": nuse,
    }
    return calib_ok, notes, determinism, fresh, floor


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ca0", type=Path, required=True, help="CA0 dump dir (ca0.npz + summary.json)")
    p.add_argument("--oracle-bank", type=Path, default=None, help="c0_oracle_livebank, not phase_b_dump")
    p.add_argument("--ckpt", default="hf_pusht")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-det-frames", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dump-only", action="store_true", help="skip GPU (tests / no model)")
    args = p.parse_args(argv)

    th_all = load_thresholds()
    th = th_all.get("encoder_floor") or {}
    ca0, summary = load_ca0(args.ca0)
    dump = dump_metrics(ca0, summary)

    calib_ok = True
    calib_notes: list[str] = []
    determinism: dict = {}
    fresh: dict = {}
    floor = 0.0
    units = {
        "tensor": "phase_b.encode_frames → encode({'pixels'})['emb'][:, 0]",
        "norm": "L2 (numpy / torch.linalg.vector_norm)",
        "normalization": "none (raw encoder embedding, no SIGReg reshape at eval)",
        "same_space_as_ca0": True,
    }

    fork = dump["fork_mean_d_end_m1"]
    reported = dump["fork_reported_d_end_m1"]
    xcheck_d_end = True
    if reported is not None and _rel_agree(fork, reported) > 1e-3:
        calib_ok = False
        xcheck_d_end = False
        calib_notes.append("dump d_end m=1 disagrees with summary.json")

    if not args.dump_only:
        gpu_ok, gpu_notes, determinism, fresh, floor = _run_gpu(args, ca0, dump, th)
        calib_ok = calib_ok and gpu_ok
        calib_notes.extend(gpu_notes)
    else:
        calib_notes.append("dump-only: skipped GPU determinism / fresh P")
        determinism = {"same_state_reencode_median": 0.0, "skipped": True}

    onestep = dump["onestep_median"]
    null = dump["adjacent_median"]
    frac = bracket_frac(onestep, floor, null)
    adj_into = np.asarray(dump["adjacent_into_next"], dtype=np.float64)
    err = np.asarray(dump["onestep"], dtype=np.float64)
    n = min(len(err), len(adj_into))
    step_frac = []
    for e, a in zip(err[:n], adj_into[:n]):
        f = bracket_frac(float(e), floor, float(a))
        if f is not None:
            step_frac.append(f)
    step_frac = np.asarray(step_frac, dtype=np.float64) if step_frac else np.zeros(0)
    decision = decide_guard(frac, calib_ok=calib_ok, th=th)
    rel = float(onestep / max(null, 1e-8))
    t_h = 5
    payload = {
        "calib_ok": calib_ok,
        "calib_notes": calib_notes,
        "units": units,
        "perfect_floor": floor,
        "same_state_reencode_median": float(determinism.get("same_state_reencode_median", floor)),
        "determinism": determinism,
        "fresh_P": fresh,
        "dump_vs_fresh_P": fresh.get("dump_vs_fresh_rel"),
        "fork_metric": {
            "name": "mean ‖ẑ_end−z*‖ at m=1",
            "value": dump["fork_mean_d_end_m1"],
            "summary_json": dump["fork_reported_d_end_m1"],
            "mean_last_true_vs_zstar": dump["mean_last_true_vs_zstar"],
            "note": "This is CA0's 1.43. It is distance-to-goal, not one-step residual.",
        },
        "bracket_metric": {
            "name": "median ‖ẑ_{t+1}−z_{t+1}‖ (m=1 teacher-force, t≥history)",
            "onestep_median": dump["onestep_median"],
            "onestep_mean": dump["onestep_mean"],
            "onestep_dist": {
                "mean": dump["onestep_mean"],
                "median": dump["onestep_median"],
                **_pcts(err),
            },
            "adjacent_median": dump["adjacent_median"],
            "adjacent_mean": dump["adjacent_mean"],
            "perfect_floor": floor,
            "null_identity": dump["adjacent_median"],
            "random_pair_spread_median": dump["random_pair_spread_median"],
            "frac": frac,
            "frac_from_medians": frac,
            "frac_per_step_median": float(np.median(step_frac)) if step_frac.size else None,
            "frac_per_step_mean": float(step_frac.mean()) if step_frac.size else None,
            "frac_per_step_dist": _pcts(step_frac),
            "n_steps": int(len(err)),
        },
        "frac": frac,
        "xcheck_reproduces_d_end": xcheck_d_end,
        "xcheck_onestep_vs_d_end": {
            "onestep_median": dump["onestep_median"],
            "fork_mean_d_end_m1": dump["fork_mean_d_end_m1"],
            "same": _rel_agree(dump["onestep_median"], dump["fork_mean_d_end_m1"]) < 0.05,
            "note": "Must stay two columns. frac uses one-step, not d_end.",
        },
        "compounding_descriptive": {
            "onestep_over_adjacent": rel,
            "naive_linear_T5": rel * t_h,
            "naive_sqrt_T5": rel * float(np.sqrt(t_h)),
            "note": "Not a gate. Independent isotropic errors would look more like sqrt(T); a bias more like T.",
        },
        "thresholds": th,
        "guard_decision": decision,
        "gate_part_b": decision in ("PARTIAL", "INFIDELITY"),
        "n_pairs": dump["n_pairs"],
        "ckpt": args.ckpt,
        "ca0": str(args.ca0),
        "oracle_bank": str(args.oracle_bank) if args.oracle_bank else None,
    }
    out = Path(args.out)
    if out.suffix == ".json":
        out.parent.mkdir(parents=True, exist_ok=True)
        path = out
    else:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "encoder_floor.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    ca0_dir = Path(args.ca0)
    if ca0_dir.is_dir():
        shutil.copy(path, ca0_dir / "encoder_floor.json")
    preview = {
        k: payload[k]
        for k in (
            "calib_ok",
            "guard_decision",
            "gate_part_b",
            "frac",
            "perfect_floor",
            "fork_metric",
            "bracket_metric",
            "xcheck_reproduces_d_end",
            "xcheck_onestep_vs_d_end",
            "calib_notes",
        )
    }
    print(json.dumps(preview, indent=2, default=str))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
