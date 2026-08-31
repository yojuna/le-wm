#!/usr/bin/env python3
"""B1: linear vs MLP probes per state factor + one causal intervention.

  python scripts/latent_probe.py --dump eval_results/pusht/phase_b_dump/seed0/dump.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

from phase_b import HISTORY, effective_rank, factor_names_for_env, load_dump  # noqa: E402


def _flatten_holdout(dump: dict, val_frac: float, seed: int):
    z = dump["z"]  # (N, L, D)
    state = dump["state"]
    k = dump["remaining_k"]
    pose = dump["remaining_pose"]
    groups = dump["episode_id"].astype(np.int32)
    n, l, d = z.shape
    z_flat = z.reshape(n * l, d)
    extra = {
        "remaining_k": k.reshape(n * l),
        "remaining_pose": pose.reshape(n * l),
    }
    y = state.reshape(n * l, state.shape[-1])
    g = np.repeat(groups, l)
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    if len(uniq) == 1:
        val_eps = uniq
        train_eps = uniq
        same_ep = True
    else:
        val_eps = set(uniq[:n_val].tolist())
        train_eps = set(uniq[n_val:].tolist()) or set(uniq[n_val - 1 :].tolist())
        # keep at least one train ep
        if not train_eps:
            train_eps = {int(uniq[-1])}
            val_eps -= train_eps
        same_ep = False
    train_m = np.isin(g, list(train_eps))
    val_m = np.isin(g, list(val_eps))
    return {
        "z_train": z_flat[train_m],
        "z_val": z_flat[val_m],
        "y_train": y[train_m],
        "y_val": y[val_m],
        "k_train": extra["remaining_k"][train_m],
        "k_val": extra["remaining_k"][val_m],
        "pose_train": extra["remaining_pose"][train_m],
        "pose_val": extra["remaining_pose"][val_m],
        "same_episode_fallback": same_ep,
        "n_train_eps": len(train_eps),
        "n_val_eps": len(val_eps),
    }


def _fit_r2(x_tr, y_tr, x_va, y_va, kind: str, seed: int) -> float:
    y_tr = np.asarray(y_tr).reshape(-1)
    y_va = np.asarray(y_va).reshape(-1)
    if kind == "linear":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                max_iter=400,
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=20,
            ),
        )
    model.fit(x_tr, y_tr)
    pred = model.predict(x_va)
    return float(r2_score(y_va, pred))


def fit_linear_direction(x_tr, y_tr) -> np.ndarray:
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe.fit(x_tr, np.asarray(y_tr).reshape(-1))
    ridge: Ridge = pipe.named_steps["ridge"]
    scaler: StandardScaler = pipe.named_steps["standardscaler"]
    # dy/dz in original z space: w / scale
    w = ridge.coef_ / np.clip(scaler.scale_, 1e-8, None)
    nrm = np.linalg.norm(w)
    if nrm < 1e-12:
        return w
    return w / nrm


def intervention_hit(
    dump: dict,
    factor_index: int,
    direction: np.ndarray,
    *,
    eps: float,
    n_trials: int,
    seed: int,
) -> dict:
    """One-step P: compare probe(next|z) vs probe(next|z+eps d) with fixed actions.

    Uses z_hat[:, HISTORY] as the model's one-step prediction from true history
    (already in the dump). For the perturbed branch we approximate by adding
    eps * direction to z at the last history frame and reading the linear
    probe on z_hat — true causal P needs the live model. If --model-ckpt is
    not used, we score whether the *encoded* z already moves the probe
    (sufficiency), and whether z_hat tracks that direction (weak P check):
    corr(probe(z_hat_t), probe(z_t)) along the path.
    """
    z = dump["z"]
    z_hat = dump["z_hat"]
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    idx = rng.choice(n, size=min(n_trials, n), replace=False)
    w = direction.reshape(1, -1)
    hits = []
    deltas = []
    for i in idx:
        z0 = z[i, HISTORY - 1]
        pred0 = float(z_hat[i, HISTORY] @ w.T)
        pred1 = float((z_hat[i, HISTORY] + eps * direction) @ w.T)
        # Without a live P call, the first-order check is: probe on z itself
        # increases when we add eps * d (true by construction for linear probe).
        # The useful check stored here: does imagined next already align?
        base = float(z0 @ w.T)
        nxt = float(z_hat[i, HISTORY] @ w.T)
        # Sign of imagined step along the factor vs sign of real z step
        real = float(z[i, HISTORY] @ w.T) - base
        imag = nxt - base
        hit = bool(real * imag > 0) or (abs(real) < 1e-6 and abs(imag) < 1e-6)
        hits.append(hit)
        deltas.append(imag - real)
    # Construction check: adding eps * d to z increases probe
    construct = float((z[idx[0], HISTORY - 1] + eps * direction) @ w.T) > float(
        z[idx[0], HISTORY - 1] @ w.T
    )
    return {
        "factor_index": int(factor_index),
        "eps": float(eps),
        "n_trials": int(len(idx)),
        "imagined_step_sign_hit": float(np.mean(hits)),
        "mean_imag_minus_real": float(np.mean(deltas)),
        "probe_increases_on_plus_eps": bool(construct),
        "note": (
            "Sign hit: imagined one-step Δprobe matches real Δprobe. "
            "Live P(z+eps d) is in latent_probe.intervene_live()."
        ),
    }


def intervene_live(
    dump: dict,
    factor_index: int,
    direction: np.ndarray,
    *,
    env: str,
    eps: float,
    n_trials: int,
    seed: int,
    device: str,
) -> dict | None:
    """Optional: load frozen trunk and run one P step with z vs z+eps d."""
    try:
        import torch
        from eval_live import ENV_REGISTRY
        from eval_setup import load_lewm_checkpoint
        from phase_b import pad_action, HISTORY as HS
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}

    spec = ENV_REGISTRY[env]
    model = load_lewm_checkpoint(spec.ckpt_dir)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    z = dump["z"]
    rng = np.random.default_rng(seed)
    n = z.shape[0]
    Lhist = HS
    if z.shape[1] < Lhist + 1:
        return {"error": "segment shorter than history+1"}
    idx = rng.choice(n, size=min(n_trials, n), replace=False)
    w = torch.from_numpy(direction.astype(np.float32)).to(device)
    hits = []
    with torch.no_grad():
        for i in idx:
            hist = torch.from_numpy(z[i, :Lhist]).unsqueeze(0).to(device)
            act = torch.zeros(1, Lhist, 10, device=device)
            act_emb = model.action_encoder(act)
            pred0 = model.predict(hist, act_emb)[:, -1, :]
            hist_p = hist.clone()
            hist_p[:, -1, :] = hist_p[:, -1, :] + eps * w
            pred1 = model.predict(hist_p, act_emb)[:, -1, :]
            d0 = float((pred0[0] * w).sum())
            d1 = float((pred1[0] * w).sum())
            hits.append(d1 > d0)
    return {
        "factor_index": int(factor_index),
        "eps": float(eps),
        "n_trials": int(len(idx)),
        "hit_rate": float(np.mean(hits)) if hits else float("nan"),
        "device": device,
    }


def phi_project_dump(dump: dict, phi_weights: Path) -> np.ndarray:
    """u = φ(z) using a saved ReachabilityHead (v2 reach.pt)."""
    import torch
    from reachability import ReachabilityHead

    blob = torch.load(phi_weights, map_location="cpu", weights_only=True)
    meta: dict = {}
    weight_state = blob
    if isinstance(blob, dict) and "reach" in blob:
        weight_state = blob["reach"]
        meta = blob.get("meta") or {}
    head = ReachabilityHead(
        input_dim=int(meta.get("input_dim", 192)),
        hidden_dim=int(meta.get("hidden_dim", 256)),
        output_dim=int(meta.get("output_dim", 64)),
        distance_mode=str(meta.get("distance_mode", "euclidean")),
        iqe_k=int(meta.get("iqe_k", 8)),
        iqe_l=int(meta.get("iqe_l", 8)),
    )
    head.load_state_dict(weight_state)
    head.eval()
    z = dump["z"]
    flat = torch.from_numpy(z.reshape(-1, z.shape[-1]).astype(np.float32))
    with torch.no_grad():
        return head.project(flat, detach_z=True).cpu().numpy()


def rank_report(dump: dict, phi_weights: Path | None) -> dict:
    z = dump["z"]
    flat_z = z.reshape(-1, z.shape[-1])
    out = {
        "z_shape": list(z.shape),
        "effective_rank_z": effective_rank(flat_z),
        "z_dim": int(flat_z.shape[-1]),
        "n_tokens": int(flat_z.shape[0]),
    }
    if phi_weights is not None and Path(phi_weights).exists():
        u = phi_project_dump(dump, Path(phi_weights))
        out["effective_rank_u"] = effective_rank(u)
        out["u_dim"] = int(u.shape[-1])
        out["rank_u_over_rank_z"] = (
            out["effective_rank_u"] / max(out["effective_rank_z"], 1e-6)
        )
        out["phi_weights"] = str(phi_weights)
    else:
        out["effective_rank_u"] = None
        out["note_u"] = f"phi weights missing at {phi_weights}"
    return out


def plot_r2(rows: list[dict], out: Path) -> None:
    names = [r["name"] for r in rows]
    lin = [r["linear_r2"] for r in rows]
    mlp = [r["mlp_r2"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(names)), 4))
    ax.bar(x - 0.18, lin, width=0.36, label="linear (Ridge)")
    ax.bar(x + 0.18, mlp, width=0.36, label="MLP")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("held-out R²")
    ax.set_title("Frozen-z probes (episode holdout)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-mlp", action="store_true")
    p.add_argument("--intervene-live", action="store_true")
    p.add_argument("--effective-rank", action="store_true", help="C0.4 participation ratio of z vs φ(z)")
    p.add_argument(
        "--phi-weights",
        default="",
        help="reach.pt for u=φ(z); default stablewm/checkpoints/pusht/lewm_phi_v2/reach.pt",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)

    dump = load_dump(args.dump)
    meta = dump.get("meta") or {}
    env = str(meta.get("env", "pusht"))
    names = [str(x) for x in dump["factor_names"].tolist()]
    if env == "reacher" or any(n.startswith("factor_") for n in names):
        names = list(factor_names_for_env(env, dump["state"].shape[-1]))
    split = _flatten_holdout(dump, args.val_frac, args.seed)

    rows = []
    extras = [
        ("remaining_k", split["k_train"], split["k_val"], True),
        ("remaining_pose", split["pose_train"], split["pose_val"], False),
    ]
    for i, name in enumerate(names):
        y_tr = split["y_train"][:, i]
        y_va = split["y_val"][:, i]
        lin = _fit_r2(split["z_train"], y_tr, split["z_val"], y_va, "linear", args.seed)
        mlp = (
            float("nan")
            if args.skip_mlp
            else _fit_r2(split["z_train"], y_tr, split["z_val"], y_va, "mlp", args.seed)
        )
        rows.append(
            {
                "name": name,
                "kind": "state",
                "sanity_only": False,
                "linear_r2": lin,
                "mlp_r2": mlp,
                "gap_mlp_minus_linear": (
                    float(mlp - lin) if mlp == mlp else float("nan")
                ),
            }
        )
    for name, y_tr, y_va, sanity in extras:
        lin = _fit_r2(split["z_train"], y_tr, split["z_val"], y_va, "linear", args.seed)
        mlp = (
            float("nan")
            if args.skip_mlp
            else _fit_r2(split["z_train"], y_tr, split["z_val"], y_va, "mlp", args.seed)
        )
        rows.append(
            {
                "name": name,
                "kind": "path",
                "sanity_only": sanity,
                "linear_r2": lin,
                "mlp_r2": mlp,
                "gap_mlp_minus_linear": (
                    float(mlp - lin) if mlp == mlp else float("nan")
                ),
            }
        )

    # Intervention target: block_x on PushT, else first factor
    target = "block_x" if "block_x" in names else names[0]
    t_i = names.index(target)
    direction = fit_linear_direction(split["z_train"], split["y_train"][:, t_i])
    offline = intervention_hit(
        dump, t_i, direction, eps=0.5, n_trials=24, seed=args.seed
    )
    live = None
    if args.intervene_live:
        live = intervene_live(
            dump,
            t_i,
            direction,
            env=env,
            eps=0.5,
            n_trials=16,
            seed=args.seed,
            device=args.device,
        )

    state_rows = [r for r in rows if r["kind"] == "state"]
    mean_lin = float(np.nanmean([r["linear_r2"] for r in state_rows]))
    mean_mlp = float(np.nanmean([r["mlp_r2"] for r in state_rows]))
    # D6 recommendation (not a flip — recorded after the run)
    if mean_lin >= 0.4:
        d6 = "keep_extract"
        d6_reason = f"mean linear R²={mean_lin:.3f} on state factors"
    elif mean_mlp >= 0.4 and (mean_mlp - mean_lin) >= 0.2:
        d6 = "flip_candidate"
        d6_reason = (
            f"state factors present nonlinearly "
            f"(MLP {mean_mlp:.3f} vs linear {mean_lin:.3f})"
        )
    else:
        d6 = "information_may_be_absent"
        d6_reason = f"mean linear={mean_lin:.3f} MLP={mean_mlp:.3f}"

    out_dir = args.out_dir or args.dump.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_r2(rows, out_dir / "probe_r2.png")
    summary = {
        "dump": str(args.dump),
        "env": env,
        "split": {
            "n_train_eps": split["n_train_eps"],
            "n_val_eps": split["n_val_eps"],
            "same_episode_fallback": split["same_episode_fallback"],
        },
        "probes": rows,
        "mean_state_linear_r2": mean_lin,
        "mean_state_mlp_r2": mean_mlp,
        "intervention_target": target,
        "intervention_offline": offline,
        "intervention_live": live,
        "d6_recommendation": d6,
        "d6_reason": d6_reason,
        "note": (
            "remaining_k is a sanity check (time-smooth z), not the D6 gate. "
            "Flip D6 only if MLP>>linear on state factors and/or live intervention misses."
        ),
    }
    if args.effective_rank:
        phi = args.phi_weights or str(
            Path(os.environ["STABLEWM_HOME"])
            / "checkpoints"
            / "pusht"
            / "lewm_phi_v2"
            / "reach.pt"
        )
        summary["effective_rank"] = rank_report(dump, Path(phi) if phi else None)
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2))
    print_keys = [
        "env",
        "mean_state_linear_r2",
        "mean_state_mlp_r2",
        "d6_recommendation",
        "d6_reason",
        "intervention_target",
    ]
    printable = {k: summary[k] for k in print_keys}
    if "effective_rank" in summary:
        printable["effective_rank"] = summary["effective_rank"]
    print(json.dumps(printable, indent=2))
    print(f"wrote {out_dir / 'probe_summary.json'}")


if __name__ == "__main__":
    main()
