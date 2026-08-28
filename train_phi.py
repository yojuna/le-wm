#!/usr/bin/env python3
"""Train lewm-phi ReachabilityHead on frozen pretrained LeWM (Protocol T1)."""

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

from eval_setup import load_lewm_checkpoint
from phi_data import HindsightPairDataset, collate_hindsight
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
    """pixels: (B, C, H, W) -> emb (B, D) using frozen encoder+projector."""
    info = {"pixels": pixels.unsqueeze(1)}  # (B, 1, C, H, W)
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
    if denom < 1e-8:
        return float("nan")
    return float((x * y).sum() / denom)


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
            optimizer.step()

        losses.append(float(loss.item()))
        ds.append(d.detach().cpu())
        ks.append(k.detach().cpu())

    d_cat = torch.cat(ds)
    k_cat = torch.cat(ks)
    return {
        "loss": float(np.mean(losses)),
        "mean_d": float(d_cat.mean()),
        "mean_k": float(k_cat.mean()),
        "corr_d_k": pearson_corr(d_cat, k_cat),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="hf_pusht", help="checkpoint folder under STABLEWM_HOME/checkpoints")
    p.add_argument("--dataset", default="pusht_expert_train", help="dataset name (no extension)")
    p.add_argument("--k-max", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--samples-per-epoch", type=int, default=4096)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write reach.pt (default: STABLEWM_HOME/checkpoints/pusht/lewm_phi)",
    )
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cache = Path(os.environ["STABLEWM_HOME"])
    out_dir = args.out_dir or (cache / "checkpoints" / "pusht" / "lewm_phi")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"loading trunk {args.ckpt}")
    model = load_lewm_checkpoint(args.ckpt, cache_dir=cache)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    print(f"loading dataset {args.dataset}")
    dataset = swm.data.load_dataset(
        args.dataset,
        transform=None,
        cache_dir=os.environ.get("LOCAL_DATASET_DIR", None),
        keys_to_cache=["pixels"],
    )
    transform = img_transform(224)
    full = HindsightPairDataset(
        dataset,
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
        num_workers=max(1, args.num_workers // 2),
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
            torch.save({"reach": reach.state_dict(), "meta": row}, best_path)
            print(f"  saved {best_path} (val corr={best_corr:.3f})")

    meta_path = out_dir / "train_phi_meta.json"
    meta_path.write_text(json.dumps({"args": vars(args), "history": history, "best_corr": best_corr}, indent=2, default=str))
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
