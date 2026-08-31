#!/usr/bin/env python3
"""Offset autopsy Phase A — real-path ranking vs imagination gap vs cost spread.

See docs/05_offset_autopsy.md. Compares L2-z / trained Euclidean φ / random φ
on the same kinematic offset pairs (no full CEM success campaign).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
import stable_worldmodel as swm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

from eval_logging.extractors import pusht_pose_errors  # noqa: E402
from eval_logging.pairs import (  # noqa: E402
    TrajectoryBank,
    collect_kinematic_bank,
    sample_eval_pairs,
)
from eval_setup import attach_reach_head, load_lewm_checkpoint  # noqa: E402
from phi_data import frame_to_tensor  # noqa: E402
from reachability import ReachabilityHead  # noqa: E402


HISTORY = 3
ACTION_DIM = 10
GOAL_OFFSET = 25


@dataclass
class IndexedPair:
    ep_i: int
    start: int
    goal: int
    pos_progress: float


def img_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


@torch.no_grad()
def encode_frames(model, pixels: torch.Tensor) -> torch.Tensor:
    """pixels (B,C,H,W) → emb (B,D)."""
    out = model.encode({"pixels": pixels.unsqueeze(1)})
    return out["emb"][:, 0]


def pad_action(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[: min(len(a), ACTION_DIM)] = a[:ACTION_DIM]
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        order = np.argsort(v)
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        return r

    return pearson(rank(np.asarray(x)), rank(np.asarray(y)))


def frac_decreasing(d: np.ndarray) -> float:
    if len(d) < 2:
        return float("nan")
    return float(np.mean(d[1:] < d[:-1] - 1e-8))


def build_indexed_pairs(bank: TrajectoryBank, *, n_pairs: int, seed: int) -> list[IndexedPair]:
    """Reuse eval pair sampler, then recover (ep_i, start) by matching frames."""
    pairs = sample_eval_pairs(
        bank,
        num_eval=n_pairs,
        goal_offset=GOAL_OFFSET,
        seed=seed,
        prefer_success=True,
        mode="offset",
    )
    indexed: list[IndexedPair] = []
    for p in pairs:
        found = None
        for ep_i, ep in enumerate(bank.episodes):
            if ep.seed != p.seed:
                continue
            if p.start_step >= len(ep.pixels):
                continue
            if np.allclose(ep.pixels[p.start_step], p.init_pixels):
                found = (ep_i, p.start_step, p.start_step + GOAL_OFFSET)
                break
        if found is None:
            # fallback: search by start_step + seed only
            for ep_i, ep in enumerate(bank.episodes):
                if ep.seed == p.seed and p.start_step + GOAL_OFFSET < len(ep.pixels):
                    found = (ep_i, p.start_step, p.start_step + GOAL_OFFSET)
                    break
        if found is None:
            raise RuntimeError("could not re-index eval pair into bank")
        ep_i, start, goal = found
        indexed.append(
            IndexedPair(
                ep_i=ep_i, start=start, goal=goal, pos_progress=float(p.pos_progress)
            )
        )
    return indexed


@torch.no_grad()
def encode_path(
    model, ep, start: int, goal: int, transform, device
) -> tuple[torch.Tensor, np.ndarray]:
    """Return z (L,D) and remaining-steps array for frames start..goal inclusive."""
    zs = []
    for t in range(start, goal + 1):
        pix = frame_to_tensor(ep.pixels[t], transform).unsqueeze(0).to(device)
        zs.append(encode_frames(model, pix)[0].cpu())
    z = torch.stack(zs, dim=0)
    remaining = np.arange(goal - start, -1, -1, dtype=np.float64)
    return z, remaining


@torch.no_grad()
def imagine_path(
    model,
    z_true: torch.Tensor,
    ep,
    start: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Autoregressive predict from true history embeddings + true actions.

    z_true: (L, D) for frames start..goal (L = GOAL_OFFSET+1).
    Returns z_hat of shape (L, D) with first HISTORY frames = true, rest predicted.
    """
    L = z_true.size(0)
    HS = HISTORY
    z_true = z_true.to(device)
    # actions: need HS + (L - HS) entries aligned to frames; use action at frame t
    acts = []
    for t in range(start, start + L):
        if t < len(ep.action):
            acts.append(pad_action(ep.action[t]))
        else:
            acts.append(np.zeros(ACTION_DIM, np.float32))
    acts = torch.from_numpy(np.stack(acts, axis=0)).to(device)  # (L, 10)

    emb = z_true[:HS].unsqueeze(0).clone()  # (1, HS, D)
    act = acts[:HS].unsqueeze(0).clone()  # (1, HS, 10)
    out = [z_true[i].cpu() for i in range(HS)]
    n_steps = L - HS
    for t in range(n_steps):
        act_emb = model.action_encoder(act)
        pred = model.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]  # (1,1,D)
        emb = torch.cat([emb, pred], dim=1)
        next_a = acts[HS + t : HS + t + 1].unsqueeze(0)
        act = torch.cat([act, next_a], dim=1)
        out.append(pred[0, 0].cpu())
    return torch.stack(out, dim=0)


def cost_to_goal(name: str, z: torch.Tensor, z_g: torch.Tensor, reach: ReachabilityHead | None):
    """z (L,D), z_g (D,) → distances (L,)."""
    if name == "l2_z":
        zg = z_g.unsqueeze(0).expand(z.size(0), -1)
        return torch.linalg.vector_norm(z - zg, ord=2, dim=-1).cpu().numpy()
    assert reach is not None
    device = next(reach.parameters()).device
    z = z.to(device)
    zg = z_g.to(device).unsqueeze(0).expand(z.size(0), -1)
    with torch.no_grad():
        return reach.distance(z, zg, detach_z=True).cpu().numpy()


@torch.no_grad()
def candidate_cost_spread(
    model,
    reach_phi: ReachabilityHead,
    reach_rnd: ReachabilityHead,
    ep,
    start: int,
    goal: int,
    transform,
    device,
    *,
    n_samples: int,
    horizon: int,
    rng: np.random.Generator,
) -> dict:
    """Random action candidates from true history → cost mean/std per head."""
    HS = HISTORY
    # Build pixel history + goal
    pix = []
    for t in range(start - HS + 1, start + 1):
        tt = max(t, 0)
        pix.append(frame_to_tensor(ep.pixels[tt], transform))
    pixels = torch.stack(pix, dim=0).unsqueeze(0).unsqueeze(0)  # (1,1,HS,C,H,W)
    pixels = pixels.expand(1, n_samples, HS, -1, -1, -1).to(device)
    goal_pix = (
        frame_to_tensor(ep.pixels[goal], transform)
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(1, n_samples, 1, -1, -1, -1)
        .to(device)
    )
    # history actions + random future
    hist_a = []
    for t in range(start - HS + 1, start + 1):
        tt = max(t, 0)
        if tt < len(ep.action):
            hist_a.append(pad_action(ep.action[tt]))
        else:
            hist_a.append(np.zeros(ACTION_DIM, np.float32))
    hist_a = np.stack(hist_a, axis=0)  # (HS, 10)
    # action_sequence length = HS + horizon (rollout uses T-H steps + final pred)
    T = HS + horizon
    cands = np.zeros((n_samples, T, ACTION_DIM), dtype=np.float32)
    for s in range(n_samples):
        cands[s, :HS] = hist_a
        cands[s, HS:] = rng.normal(0, 0.3, size=(horizon, ACTION_DIM)).astype(
            np.float32
        )
    action_seq = torch.from_numpy(cands).unsqueeze(0).to(device)  # (1,S,T,10)

    info_base = {"pixels": pixels, "goal": goal_pix}

    # L2
    attach_reach_head(model, plan_cost="l2_z", cache_goal_emb=False, device=str(device))
    info = {k: v.clone() for k, v in info_base.items()}
    c_l2 = model.get_cost(info, action_seq.clone()).reshape(-1).float().cpu().numpy()

    model.plan_cost = "phi_d"
    model.reach = reach_phi
    model.reach.to(device).eval()
    model.cache_goal_emb = False
    model._cached_goal_emb = None
    model.clear_goal_cache()
    info = {k: v.clone() for k, v in info_base.items()}
    c_phi = model.get_cost(info, action_seq.clone()).reshape(-1).float().cpu().numpy()

    model.reach = reach_rnd
    model.reach.to(device).eval()
    model.clear_goal_cache()
    info = {k: v.clone() for k, v in info_base.items()}
    c_rnd = model.get_cost(info, action_seq.clone()).reshape(-1).float().cpu().numpy()

    def pack(name, c):
        return {
            "name": name,
            "mean": float(c.mean()),
            "std": float(c.std()),
            "min": float(c.min()),
            "max": float(c.max()),
            "range": float(c.max() - c.min()),
            "cv": float(c.std() / (abs(c.mean()) + 1e-8)),
        }

    return {"l2_z": pack("l2_z", c_l2), "phi": pack("phi", c_phi), "random": pack("random", c_rnd)}


def plot_real_progress(curves: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (rem, d) in curves.items():
        # average across pairs: rem is shared
        ax.plot(rem, d, marker="o", label=name)
    ax.set_xlabel("remaining steps to goal (real path)")
    ax.set_ylabel("mean distance to z_g")
    ax.set_title("Real-path cost vs progress (offset pairs)")
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_imagination_gap(rows: list[dict], out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=False)
    names = ["l2_z", "phi", "random"]
    for ax, name in zip(axes, names):
        real = [r[f"real_end_{name}"] for r in rows]
        imag = [r[f"imag_end_{name}"] for r in rows]
        ax.scatter(real, imag, alpha=0.7)
        lim = max(max(real), max(imag)) * 1.05 + 1e-6
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel(f"d(z_true_end, g) [{name}]")
        ax.set_ylabel(f"d(z_hat_end, g) [{name}]")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Imagination gap at path end (true actions through predictor)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_cost_spread(spreads: list[dict], out: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["l2_z", "phi", "random"]
    means = []
    stds = []
    for lab in labels:
        vals_std = [s[lab]["std"] for s in spreads]
        vals_mean = [s[lab]["mean"] for s in spreads]
        means.append(float(np.mean(vals_mean)))
        stds.append(float(np.mean(vals_std)))
    x = np.arange(len(labels))
    ax.bar(x - 0.15, means, width=0.3, label="mean cost")
    ax.bar(x + 0.15, stds, width=0.3, label="std across candidates")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("CEM-like candidate cost spread (random actions)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def map_hypotheses(summary: dict) -> dict:
    """Heuristic H1–H4 labels from aggregate stats."""
    real = summary["real_path"]
    imag = summary["imagination"]
    spread = summary["spread"]

    phi_spear = real["phi"]["spearman_mean"]
    l2_spear = real["l2_z"]["spearman_mean"]
    rnd_spear = real["random"]["spearman_mean"]

    # H3: real ranking weak for phi
    h3 = phi_spear < 0.35

    # H1: real OK but imagination blows up relative error
    gap_phi = imag["phi"]["rel_gap_mean"]
    gap_l2 = imag["l2_z"]["rel_gap_mean"]
    h1 = (phi_spear >= 0.5) and (gap_phi > 0.25)

    # H2: phi cost std much smaller than l2 (collapsed landscape)
    std_phi = spread["phi"]["std_mean"]
    std_l2 = spread["l2_z"]["std_mean"]
    h2 = std_phi < 0.35 * std_l2

    # H4: all spears mediocre
    h4 = max(phi_spear, l2_spear, rnd_spear) < 0.4

    votes = {
        "H1_imagination_ood": bool(h1),
        "H2_collapsed_spread": bool(h2),
        "H3_weak_real_ranking": bool(h3),
        "H4_all_costs_weak": bool(h4),
    }
    # primary: first true in priority H1, H3, H2, H4
    primary = "inconclusive"
    for key in (
        "H1_imagination_ood",
        "H3_weak_real_ranking",
        "H2_collapsed_spread",
        "H4_all_costs_weak",
    ):
        if votes[key]:
            primary = key
            break
    if not any(votes.values()):
        # soft: compare gap vs spear
        if gap_phi > gap_l2 and phi_spear >= l2_spear:
            primary = "H1_imagination_ood_soft"
            votes["H1_imagination_ood"] = True
        elif phi_spear >= 0.5:
            primary = "real_ranking_ok_check_transfer"
    votes["primary"] = primary
    votes["notes"] = {
        "phi_spearman": phi_spear,
        "l2_spearman": l2_spear,
        "random_spearman": rnd_spear,
        "phi_rel_gap": gap_phi,
        "l2_rel_gap": gap_l2,
        "phi_cost_std": std_phi,
        "l2_cost_std": std_l2,
    }
    return votes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="hf_pusht")
    p.add_argument(
        "--phi-weights",
        type=Path,
        default=None,
        help="default: stablewm/.../lewm_phi_v2/reach.pt",
    )
    p.add_argument("--collect-episodes", type=int, default=200)
    p.add_argument("--n-pairs", type=int, default=24)
    p.add_argument("--n-spread-pairs", type=int, default=8)
    p.add_argument("--n-candidates", type=int, default=64)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    args = p.parse_args()

    cache = Path(os.environ["STABLEWM_HOME"])
    phi_path = args.phi_weights or (
        cache / "checkpoints" / "pusht" / "lewm_phi_v2" / "reach.pt"
    )
    out_dir = args.out_dir or (
        ROOT / "eval_results" / "pusht" / "offset_autopsy"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        if args.allow_cpu:
            device = torch.device("cpu")
        else:
            raise SystemExit("CUDA unavailable; pass --allow-cpu")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)

    print(f"device={device}  phi={phi_path}")
    model = load_lewm_checkpoint(args.ckpt, cache_dir=cache)
    model.to(device).eval()
    model.requires_grad_(False)

    blob = torch.load(phi_path, map_location="cpu", weights_only=False)
    reach_phi = ReachabilityHead(distance_mode="euclidean", output_dim=64)
    reach_phi.load_state_dict(blob["reach"] if "reach" in blob else blob)
    reach_phi.to(device).eval()

    torch.manual_seed(args.seed + 123)
    reach_rnd = ReachabilityHead(distance_mode="euclidean", output_dim=64)
    reach_rnd.to(device).eval()

    print(f"collecting kinematic bank eps={args.collect_episodes}")
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=1,
        max_episode_steps=100,
        image_shape=(224, 224),
    )
    try:
        bank = collect_kinematic_bank(
            world,
            num_episodes=args.collect_episodes,
            seed=args.seed,
            env_name="swm/PushT-v1",
            horizon=80,
        )
    finally:
        world.close()
    print(f"bank episodes={len(bank.episodes)} steps={bank.num_steps}")

    indexed = build_indexed_pairs(bank, n_pairs=args.n_pairs, seed=args.seed)
    transform = img_transform(224)
    rng = np.random.default_rng(args.seed)

    # --- Real path + imagination ---
    per_pair = []
    agg_d = {
        "l2_z": [],
        "phi": [],
        "random": [],
    }  # list of (remaining, d) per pair for averaging
    spear_stats = {k: [] for k in agg_d}
    dec_stats = {k: [] for k in agg_d}
    gap_stats = {k: [] for k in agg_d}
    z_err = []

    for i, ip in enumerate(indexed):
        ep = bank.episodes[ip.ep_i]
        z, remaining = encode_path(model, ep, ip.start, ip.goal, transform, device)
        z_g = z[-1]
        z_hat = imagine_path(model, z, ep, ip.start, device=device)

        costs_real = {
            "l2_z": cost_to_goal("l2_z", z, z_g, None),
            "phi": cost_to_goal("phi", z, z_g, reach_phi),
            "random": cost_to_goal("phi", z, z_g, reach_rnd),
        }
        costs_imag = {
            "l2_z": cost_to_goal("l2_z", z_hat, z_g, None),
            "phi": cost_to_goal("phi", z_hat, z_g, reach_phi),
            "random": cost_to_goal("phi", z_hat, z_g, reach_rnd),
        }
        err = float(torch.linalg.vector_norm(z_hat - z, ord=2, dim=-1).mean())
        z_err.append(err)

        row = {
            "pair": i,
            "ep_i": ip.ep_i,
            "start": ip.start,
            "goal": ip.goal,
            "pos_progress": ip.pos_progress,
            "mean_z_rollout_err": err,
        }
        for name in ("l2_z", "phi", "random"):
            d = costs_real[name]
            di = costs_imag[name]
            sp = spearman(d, remaining)
            pe = pearson(d, remaining)
            fd = frac_decreasing(d)
            spear_stats[name].append(sp)
            dec_stats[name].append(fd)
            # relative gap at end: |d_hat - d_true| / (d_true + eps)
            rel = abs(float(di[-1]) - float(d[-1])) / (abs(float(d[-1])) + 1e-6)
            gap_stats[name].append(rel)
            row[f"spearman_{name}"] = sp
            row[f"pearson_{name}"] = pe
            row[f"frac_dec_{name}"] = fd
            row[f"real_end_{name}"] = float(d[-1])
            row[f"imag_end_{name}"] = float(di[-1])
            row[f"rel_gap_end_{name}"] = rel
            agg_d[name].append(d)
        per_pair.append(row)
        print(
            f"pair {i+1}/{len(indexed)}  φ_spear={row['spearman_phi']:.3f}  "
            f"l2_spear={row['spearman_l2_z']:.3f}  φ_rel_gap={row['rel_gap_end_phi']:.3f}  "
            f"z_err={err:.3f}"
        )

    # mean curves
    rem_axis = np.arange(GOAL_OFFSET, -1, -1, dtype=np.float64)
    mean_curves = {}
    for name, mats in agg_d.items():
        mean_curves[name] = (rem_axis, np.mean(np.stack(mats, 0), axis=0))

    plot_real_progress(mean_curves, out_dir / "real_progress.png")
    plot_imagination_gap(per_pair, out_dir / "imagination_gap.png")

    # --- Cost spread on subset ---
    spreads = []
    for ip in indexed[: args.n_spread_pairs]:
        ep = bank.episodes[ip.ep_i]
        # need history before start
        if ip.start < HISTORY:
            continue
        sp = candidate_cost_spread(
            model,
            reach_phi,
            reach_rnd,
            ep,
            ip.start,
            ip.goal,
            transform,
            device,
            n_samples=args.n_candidates,
            horizon=args.horizon,
            rng=rng,
        )
        spreads.append(sp)
        print(
            f"spread ep={ip.ep_i} start={ip.start}  "
            f"std l2={sp['l2_z']['std']:.3f} φ={sp['phi']['std']:.3f} rnd={sp['random']['std']:.3f}"
        )

    if spreads:
        plot_cost_spread(spreads, out_dir / "cost_spread.png")

    def mean_std(xs):
        xs = np.asarray(xs, dtype=np.float64)
        xs = xs[np.isfinite(xs)]
        return float(xs.mean()) if xs.size else float("nan"), float(
            xs.std(ddof=1) if xs.size > 1 else 0.0
        )

    summary = {
        "n_pairs": len(indexed),
        "goal_offset": GOAL_OFFSET,
        "phi_weights": str(phi_path),
        "seed": args.seed,
        "real_path": {},
        "imagination": {},
        "spread": {},
        "mean_z_rollout_err": float(np.mean(z_err)) if z_err else float("nan"),
    }
    for name in ("l2_z", "phi", "random"):
        ms, ss = mean_std(spear_stats[name])
        md, sd = mean_std(dec_stats[name])
        mg, sg = mean_std(gap_stats[name])
        summary["real_path"][name] = {
            "spearman_mean": ms,
            "spearman_std": ss,
            "frac_decreasing_mean": md,
        }
        summary["imagination"][name] = {"rel_gap_mean": mg, "rel_gap_std": sg}

    for name in ("l2_z", "phi", "random"):
        if spreads:
            stds = [s[name]["std"] for s in spreads]
            means = [s[name]["mean"] for s in spreads]
            cvs = [s[name]["cv"] for s in spreads]
            summary["spread"][name] = {
                "std_mean": float(np.mean(stds)),
                "mean_mean": float(np.mean(means)),
                "cv_mean": float(np.mean(cvs)),
            }
        else:
            summary["spread"][name] = {
                "std_mean": float("nan"),
                "mean_mean": float("nan"),
                "cv_mean": float("nan"),
            }

    summary["hypotheses"] = map_hypotheses(summary)
    summary["per_pair"] = per_pair
    summary["spreads_raw"] = spreads

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # markdown
    h = summary["hypotheses"]
    md = [
        "# Offset autopsy Phase A — results",
        "",
        f"**Seed:** {args.seed} · **pairs:** {len(indexed)} · **offset:** {GOAL_OFFSET}",
        f"**φ weights:** `{phi_path}`",
        "",
        "## Hypothesis call",
        "",
        f"**Primary:** `{h['primary']}`",
        "",
        "| Flag | Value |",
        "|------|-------|",
    ]
    for k, v in h.items():
        if k in ("primary", "notes"):
            continue
        md.append(f"| {k} | {v} |")
    md += [
        "",
        "### Notes",
        "",
        "```json",
        json.dumps(h["notes"], indent=2),
        "```",
        "",
        "## Real-path ranking (cost vs remaining steps)",
        "",
        "| Cost | Spearman (mean±std) | frac decreasing |",
        "|------|---------------------|-----------------|",
    ]
    for name in ("l2_z", "phi", "random"):
        r = summary["real_path"][name]
        md.append(
            f"| {name} | {r['spearman_mean']:.3f}±{r['spearman_std']:.3f} | "
            f"{r['frac_decreasing_mean']:.3f} |"
        )
    md += [
        "",
        "## Imagination gap (end of path, true actions)",
        "",
        f"Mean ‖ẑ−z‖₂ along path: **{summary['mean_z_rollout_err']:.3f}**",
        "",
        "| Cost | relative end-gap mean±std |",
        "|------|---------------------------|",
    ]
    for name in ("l2_z", "phi", "random"):
        r = summary["imagination"][name]
        md.append(
            f"| {name} | {r['rel_gap_mean']:.3f}±{r['rel_gap_std']:.3f} |"
        )
    md += [
        "",
        "## Candidate cost spread",
        "",
        "| Cost | mean | std | cv |",
        "|------|------|-----|----|",
    ]
    for name in ("l2_z", "phi", "random"):
        r = summary["spread"][name]
        md.append(
            f"| {name} | {r['mean_mean']:.3f} | {r['std_mean']:.3f} | {r['cv_mean']:.3f} |"
        )
    md += [
        "",
        "## Artifacts",
        "",
        f"- `{out_dir / 'real_progress.png'}`",
        f"- `{out_dir / 'imagination_gap.png'}`",
        f"- `{out_dir / 'cost_spread.png'}`",
        f"- `{out_dir / 'summary.json'}`",
        "",
    ]
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md))
    docs = ROOT.parent / "docs" / "lewm_phi_offset_autopsy_summary.md"
    docs.write_text("\n".join(md))
    print(f"wrote {md_path}")
    print(f"wrote {docs}")
    print(f"PRIMARY HYPOTHESIS: {h['primary']}")


if __name__ == "__main__":
    main()
