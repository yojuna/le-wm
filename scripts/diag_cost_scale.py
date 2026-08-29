#!/usr/bin/env python3
"""Compare CEM cost dynamic range for l2_z vs phi_d on the same candidates.

CEM in stable-worldmodel uses hard top-k (largest=False), so *affine* rescaling
of a single cost does not change elites. What *does* break planning is a
collapsed cost (near-zero std across samples). This script reports that.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

from eval_setup import attach_reach_head, load_lewm_checkpoint  # noqa: E402


def _stats(name: str, cost: torch.Tensor) -> dict:
    c = cost.detach().float().reshape(-1)
    return {
        "name": name,
        "shape": list(cost.shape),
        "mean": float(c.mean()),
        "std": float(c.std(unbiased=False)),
        "min": float(c.min()),
        "max": float(c.max()),
        "range": float(c.max() - c.min()),
        "cv": float(c.std(unbiased=False) / (c.mean().abs() + 1e-8)),
        "frac_near_min": float((c <= c.min() + 1e-5).float().mean()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="hf_pusht")
    p.add_argument("--phi-weights", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--history", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    cache = Path(os.environ["STABLEWM_HOME"])
    model = load_lewm_checkpoint(args.ckpt, cache_dir=cache)
    model.to(device).eval()

    # Synthetic infos matching CEM-expanded PushT shapes:
    # pixels/goal/action: (B, S, T, ...) ; candidates: (B, S, horizon, A)
    B, Hw, C, S = 1, 224, 3, args.num_samples
    hist = args.history
    plan_t = max(args.horizon, hist + 1)
    pixels = torch.randn(B, S, hist, C, Hw, Hw, device=device) * 0.5
    goal = torch.randn(B, S, 1, C, Hw, Hw, device=device) * 0.5
    action_dim = 10  # PushT LeWM Embedder input_dim (padded action)
    action = torch.randn(B, S, hist, action_dim, device=device) * 0.1
    candidates = torch.randn(B, S, plan_t, action_dim, device=device) * 0.5

    base_info = {
        "pixels": pixels,
        "goal": goal,
        "action": action,
    }

    results = []
    variants = [("l2_z", None, "l2_z"), ("phi_d", None, "phi_d_random")]
    if args.phi_weights is not None:
        variants.append(("phi_d", args.phi_weights, "phi_d_trained"))

    for plan_cost, weights, label in variants:
        attach_reach_head(
            model,
            plan_cost=plan_cost,
            phi_weights=weights,
            cache_goal_emb=True,
            device=str(device),
        )
        model.clear_goal_cache()
        model._forced_goal_cache_key = f"diag:{label}"
        info = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base_info.items()}
        with torch.no_grad():
            cost = model.get_cost(info, candidates)
        st = _stats(label, cost)
        results.append(st)
        print(
            f"{label:16s}  mean={st['mean']:.4f}  std={st['std']:.4f}  "
            f"range={st['range']:.4f}  cv={st['cv']:.4f}"
        )

    # Rank agreement: does trained φ order candidates like L2?
    if len(results) >= 2:
        attach_reach_head(model, plan_cost="l2_z", cache_goal_emb=True, device=str(device))
        model.clear_goal_cache()
        model._forced_goal_cache_key = "diag:spearman_l2"
        info = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base_info.items()}
        with torch.no_grad():
            c_l2 = model.get_cost(info, candidates).reshape(-1).cpu()

        if args.phi_weights and args.phi_weights.exists():
            attach_reach_head(
                model,
                plan_cost="phi_d",
                phi_weights=args.phi_weights,
                cache_goal_emb=True,
                device=str(device),
            )
            model.clear_goal_cache()
            model._forced_goal_cache_key = "diag:spearman_phi"
            info = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base_info.items()}
            with torch.no_grad():
                c_phi = model.get_cost(info, candidates).reshape(-1).cpu()
            # Spearman via rank corr
            r_l2 = c_l2.argsort().argsort().float()
            r_phi = c_phi.argsort().argsort().float()
            spearman = float(
                torch.corrcoef(torch.stack([r_l2, r_phi]))[0, 1].item()
            )
            print(f"spearman_rank(l2_z, phi_d_trained)={spearman:.4f}")
            results.append({"name": "spearman_l2_vs_phi", "value": spearman})

    collapsed = [
        r["name"]
        for r in results
        if isinstance(r, dict) and "std" in r and r["std"] < 1e-4
    ]
    if collapsed:
        print(f"WARNING: near-collapsed costs: {collapsed}")
    else:
        print("OK: all cost variants have non-trivial std across CEM samples")

    out = {
        "device": str(device),
        "num_samples": args.num_samples,
        "phi_weights": str(args.phi_weights) if args.phi_weights else None,
        "results": results,
    }
    out_path = args.out or (
        cache / "checkpoints" / "pusht" / "lewm_phi" / "cost_scale_diag.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
