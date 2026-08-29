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
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt
import stable_worldmodel as swm

from eval_logging.pairs import (
    TrajectoryBank,
    collect_kinematic_bank,
    collect_trajectory_bank,
)
from eval_setup import load_lewm_checkpoint
from phi_data import (
    build_train_val_datasets,
    collate_hindsight,
    sample_hindsight_pairs,
)
from reachability import ReachabilityHead

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT.parent / "stablewm"
os.environ.setdefault("STABLEWM_HOME", str(DEFAULT_CACHE))


def img_transform(img_size: int = 224):
    """Expect CHW float tensor in [0, 1]; apply ImageNet normalize + resize."""
    return transforms.Compose(
        [
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


def _append_metrics_csv(path: Path, row: dict) -> None:
    import csv

    flat = {
        "epoch": row["epoch"],
        "train_loss": row["train"]["loss"],
        "train_corr": row["train"]["corr_d_k"],
        "train_mean_d": row["train"]["mean_d"],
        "train_mean_k": row["train"]["mean_k"],
        "val_loss": row["val"]["loss"],
        "val_corr": row["val"]["corr_d_k"],
        "val_mean_d": row["val"]["mean_d"],
        "val_mean_k": row["val"]["mean_k"],
    }
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            w.writeheader()
        w.writerow(flat)


def plot_training_curves(history: list[dict], out_path: Path) -> None:
    """Write loss/corr curves with matplotlib (always; no TensorBoard required)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, [r["train"]["loss"] for r in history], label="train", marker="o")
    axes[0].plot(epochs, [r["val"]["loss"] for r in history], label="val", marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("Huber loss")
    axes[0].set_title("reachability loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [r["train"]["corr_d_k"] for r in history], label="train", marker="o")
    axes[1].plot(epochs, [r["val"]["corr_d_k"] for r in history], label="val", marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Pearson corr(d, k)")
    axes[1].set_title("distance vs temporal k")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _maybe_tb_writer(log_dir: Path, enabled: bool):
    if not enabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # noqa: BLE001
        print(f"TensorBoard unavailable ({exc}); continuing with matplotlib/CSV only")
        return None
    writer = SummaryWriter(log_dir=str(log_dir / "tb"))
    print(f"TensorBoard logdir: {log_dir / 'tb'}  (tensorboard --logdir {log_dir / 'tb'})")
    return writer


def _assert_on_device(module: torch.nn.Module, device: torch.device, *, name: str) -> None:
    param = next(module.parameters(), None)
    if param is None:
        return
    if param.device.type != device.type:
        raise RuntimeError(
            f"{name} is on {param.device}, expected {device}. "
            "Training would not use the requested GPU."
        )
    if device.type == "cuda" and device.index is not None and param.device.index not in (
        device.index,
        None,
    ):
        raise RuntimeError(f"{name} is on {param.device}, expected {device}")


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
    non_blocking = isinstance(device, torch.device) and device.type == "cuda"
    for batch in loader:
        pixels_t = batch["pixels_t"].to(device, non_blocking=non_blocking)
        pixels_tk = batch["pixels_tk"].to(device, non_blocking=non_blocking)
        k = batch["k"].to(device, non_blocking=non_blocking)

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
    p.add_argument(
        "--device",
        default="cuda",
        help="torch device (default: cuda). Use --device cuda:0 to pin a GPU.",
    )
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit falling back to CPU if CUDA is unavailable (default: abort)",
    )
    p.add_argument(
        "--tensorboard",
        action="store_true",
        help="also log scalars with torch.utils.tensorboard if installed",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            if args.allow_cpu:
                print("WARNING: CUDA unavailable; falling back to CPU (--allow-cpu)")
                device = torch.device("cpu")
            else:
                raise SystemExit(
                    "CUDA requested but torch.cuda.is_available() is False. "
                    "Fix the GPU driver/PyTorch install, or pass --allow-cpu."
                )
        # Resolve cuda -> cuda:0 so .to(device) is unambiguous
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        print(
            f"using GPU {device} ({torch.cuda.get_device_name(device)})  "
            f"mem={torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB"
        )
    else:
        print(f"using device {device}")

    cache = Path(os.environ["STABLEWM_HOME"])
    out_dir = args.out_dir or (cache / "checkpoints" / "pusht" / "lewm_phi")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    print(f"loading frozen trunk {args.ckpt}")
    model = load_lewm_checkpoint(args.ckpt, cache_dir=cache)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    _assert_on_device(model, device, name="trunk")

    print(f"collecting live PushT bank collector={args.collector}")
    bank = collect_live_bank(args)

    transform = img_transform(224)
    train_set, val_set, split_meta = build_train_val_datasets(
        bank,
        k_max=args.k_max,
        samples_per_epoch=args.samples_per_epoch,
        val_frac=args.val_frac,
        seed=args.seed,
        img_transform=transform,
    )
    print(
        f"episode split: train_eps={split_meta['n_train_episodes']} "
        f"val_eps={split_meta['n_val_episodes']} "
        f"train_pairs={split_meta['n_train_pairs']} "
        f"val_pairs={split_meta['n_val_pairs']}"
        + (
            "  WARNING: only 1 usable episode — val is not held-out"
            if split_meta.get("same_episode_fallback")
            else ""
        )
    )

    def make_loaders(train_ds):
        tr = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_hindsight,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )
        va = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_hindsight,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
        )
        return tr, va

    train_loader, val_loader = make_loaders(train_set)

    reach = ReachabilityHead(input_dim=192, output_dim=64).to(device)
    _assert_on_device(reach, device, name="reach")
    optim = torch.optim.AdamW(reach.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_corr = -1.0
    best_path = out_dir / "reach.pt"
    metrics_csv = out_dir / "metrics.csv"
    curves_png = out_dir / "training_curves.png"
    if metrics_csv.exists():
        metrics_csv.unlink()
    tb = _maybe_tb_writer(out_dir, enabled=args.tensorboard)

    for epoch in range(1, args.epochs + 1):
        # Fresh train pairs each epoch (same train episodes); val pairs stay fixed.
        if epoch > 1:
            train_set.pairs = sample_hindsight_pairs(
                train_set.episodes,
                k_max=args.k_max,
                n_samples=args.samples_per_epoch,
                seed=args.seed + epoch,
            )
            train_loader, val_loader = make_loaders(train_set)

        tr = run_epoch(model, reach, train_loader, device, train=True, optimizer=optim)
        va = run_epoch(model, reach, val_loader, device, train=False)
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        _append_metrics_csv(metrics_csv, row)
        plot_training_curves(history, curves_png)
        if tb is not None:
            tb.add_scalar("loss/train", tr["loss"], epoch)
            tb.add_scalar("loss/val", va["loss"], epoch)
            tb.add_scalar("corr/train", tr["corr_d_k"], epoch)
            tb.add_scalar("corr/val", va["corr_d_k"], epoch)
            tb.flush()
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss={tr['loss']:.4f} corr={tr['corr_d_k']:.3f}  "
            f"val_loss={va['loss']:.4f} corr={va['corr_d_k']:.3f}  "
            f"[wrote {curves_png.name}, {metrics_csv.name}]"
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
        "split": split_meta,
        "metrics_csv": str(metrics_csv),
        "training_curves": str(curves_png),
    }
    meta_path = out_dir / "train_phi_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}")
    print(f"metrics CSV: {metrics_csv}")
    print(f"curves plot: {curves_png}")
    if tb is not None:
        tb.close()


if __name__ == "__main__":
    main()
