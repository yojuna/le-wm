#!/usr/bin/env python3
"""Train lewm-phi ReachabilityHead on a frozen trunk using *live* PushT rollouts.

No HuggingFace / HDF5 expert dataset. Collects a TrajectoryBank with the same
collectors as eval_live (weak / kinematic / goal), then regresses Euclidean
φ-distance onto hindsight temporal k.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
import stable_worldmodel as swm

from eval_logging.pairs import (
    TrajectoryBank,
    collect_kinematic_bank,
    collect_trajectory_bank,
)
from eval_setup import load_lewm_checkpoint
from phi_data import LiveHindsightPairDataset, collate_hindsight
from reachability import ReachabilityHead

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT.parent / "stablewm"
os.environ.setdefault("STABLEWM_HOME", str(DEFAULT_CACHE))


def img_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


@torch.no_grad()
def encode_frames(model, pixels: torch.Tensor) -> torch.Tensor:
    """pixels: (B, C, H, W) -> emb (B, D)."""
    info = {"pixels": pixels.unsqueeze(1)}
    out = model.encode(info)
    return out["emb"][:, 0]


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) < 1e-8:
        return float("nan")
    return float((x * y).sum() / denom)


def collect_live_bank(args) -> TrajectoryBank:
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=1,
        max_episode_steps=max(args.ep_horizon + 5, 100),
        image_shape=(224, 224),
    )
    try:
        if args.collector == "kinematic":
            bank = collect_kinematic_bank(
                world,
                num_episodes=args.collect_episodes,
                seed=args.seed,
                env_name="swm/PushT-v1",
                horizon=args.ep_horizon,
            )
        else:
            bank = collect_trajectory_bank(
                world,
                num_steps=args.collect_steps,
                seed=args.seed,
                env_name="swm/PushT-v1",
                min_episode_len=args.k_max + 2,
                collector=args.collector,
            )
    finally:
        world.close()

    usable = sum(1 for ep in bank.episodes if len(ep) > args.k_max)
    print(
        f"collected bank: episodes={len(bank.episodes)} steps={bank.num_steps} "
        f"usable(>k_max)={usable} collector={bank.collector} "
        f"success_eps={bank.num_success_episodes}"
    )
    if usable == 0:
        raise RuntimeError(
            "no episodes longer than k_max; increase --collect-episodes/--collect-steps "
            "or lower --k-max"
        )
    return bank


def run_epoch(model, reach, loader, device, *, train: bool, optimizer=None):
    reach.train(train)
    losses, ds, ks = [], [], []
    for batch in loader:
        pixels_t = batch["pixels_t"].to(device)
        pixels_tk = batch["pixels_tk"].to(device)
        k = batch["k"].to(device)

        with torch.no_grad():
            z_t = encode_frames(model, pixels_t)
            z_tk = encode_frames(model, pixels_tk)

        d = reach.pairwise_distance(z_t, z_tk, detach_z=True)
        loss = F.huber_loss(d, k)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Guard: trunk must not receive grads
            for p in model.parameters():
                if p.grad is not None:
                    raise RuntimeError("trunk received gradients during train_phi")
            optimizer.step()

        losses.append(float(loss.item()))
        ds.append(d.detach().cpu())
        ks.append(k.detach().cpu())

    d_cat = torch.cat(ds)
    k_cat = torch.cat(ks)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mean_d": float(d_cat.mean()) if d_cat.numel() else float("nan"),
        "mean_k": float(k_cat.mean()) if k_cat.numel() else float("nan"),
        "corr_d_k": pearson_corr(d_cat, k_cat) if d_cat.numel() else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="hf_pusht")
    p.add_argument(
        "--collector",
        choices=["weak", "kinematic", "goal"],
        default="weak",
        help="live rollout policy for the training bank (default: weak)",
    )
    p.add_argument("--collect-episodes", type=int, default=64, help="kinematic episodes")
    p.add_argument("--collect-steps", type=int, default=8000, help="weak/goal total steps")
    p.add_argument("--ep-horizon", type=int, default=80)
    p.add_argument("--k-max", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0, help="0 avoids pickle of bank")
    p.add_argument("--samples-per-epoch", type=int, default=2048)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cache = Path(os.environ["STABLEWM_HOME"])
    out_dir = args.out_dir or (cache / "checkpoints" / "pusht" / "lewm_phi")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"loading frozen trunk {args.ckpt}")
    model = load_lewm_checkpoint(args.ckpt, cache_dir=cache)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    print(f"collecting live PushT bank collector={args.collector}")
    bank = collect_live_bank(args)

    transform = img_transform(224)
    full = LiveHindsightPairDataset(
        bank,
        k_max=args.k_max,
        img_transform=transform,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
    )
    n_val = max(1, int(len(full) * args.val_frac))
    n_train = len(full) - n_val
    train_set, val_set = random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_hindsight,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_hindsight,
        drop_last=False,
    )

    reach = ReachabilityHead(input_dim=192, output_dim=64).to(device)
    optim = torch.optim.AdamW(reach.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_corr = -1.0
    best_path = out_dir / "reach.pt"

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, reach, train_loader, device, train=True, optimizer=optim)
        va = run_epoch(model, reach, val_loader, device, train=False)
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss={tr['loss']:.4f} corr={tr['corr_d_k']:.3f}  "
            f"val_loss={va['loss']:.4f} corr={va['corr_d_k']:.3f}"
        )
        if va["corr_d_k"] == va["corr_d_k"] and va["corr_d_k"] >= best_corr:
            best_corr = va["corr_d_k"]
            torch.save(
                {
                    "reach": reach.state_dict(),
                    "meta": row,
                    "collector": args.collector,
                    "data": "live_trajectory_bank",
                },
                best_path,
            )
            print(f"  saved {best_path} (val corr={best_corr:.3f})")

    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "history": history,
        "best_corr": best_corr,
        "bank_episodes": len(bank.episodes),
        "bank_steps": bank.num_steps,
        "data_source": "live_simulator",
    }
    meta_path = out_dir / "train_phi_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
