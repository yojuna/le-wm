#!/usr/bin/env python3
"""Train Euclidean φ with imagined futures (H1 transfer fix).

See docs/06_imagined_phi.md. Loss: Huber(‖φ(z_t)−φ(ẑ_{t+k})‖, k) where ẑ
comes from frozen LeWM rollouts with true bank actions; optional real mix.
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
from phi_imagined_data import (
    HISTORY,
    build_imagined_train_val,
    collate_imagined,
    sample_imagined_hindsight_pairs,
)
from reachability import ReachabilityHead

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT.parent / "stablewm"
os.environ.setdefault("STABLEWM_HOME", str(DEFAULT_CACHE))


def img_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


@torch.no_grad()
def encode_frames(model, pixels: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) or (B,T,C,H,W) → (B,D) or (B,T,D)."""
    if pixels.ndim == 4:
        out = model.encode({"pixels": pixels.unsqueeze(1)})
        return out["emb"][:, 0]
    B, T = pixels.shape[:2]
    flat = pixels.reshape(B * T, *pixels.shape[2:])
    out = model.encode({"pixels": flat.unsqueeze(1)})
    return out["emb"][:, 0].view(B, T, -1)


@torch.no_grad()
def imagine_futures(
    model,
    z_hist: torch.Tensor,
    actions_all: torch.Tensor,
    k_int: torch.Tensor,
    *,
    history: int = HISTORY,
) -> torch.Tensor:
    """Roll out k steps; return ẑ_{t+k} (B, D).

    z_hist: (B, HS, D); actions_all: (B, HS+Kmax, A); k_int: (B,)
    """
    B, HS, D = z_hist.shape
    assert HS == history
    emb = z_hist.clone()
    act = actions_all[:, :HS].clone()
    k_max = int(k_int.max().item())
    last = emb[:, -1].clone()
    for step in range(k_max):
        act_emb = model.action_encoder(act)
        pred = model.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        next_a = actions_all[:, HS + step : HS + step + 1, :]
        act = torch.cat([act, next_a], dim=1)
        active = k_int > step  # after (step+1) preds, update if k >= step+1
        last = torch.where(active.unsqueeze(-1), pred[:, 0], last)
    return last


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
        "train_frac_imagined": row["train"]["frac_imagined"],
        "val_loss": row["val"]["loss"],
        "val_corr": row["val"]["corr_d_k"],
        "val_mean_d": row["val"]["mean_d"],
        "val_frac_imagined": row["val"]["frac_imagined"],
    }
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            w.writeheader()
        w.writerow(flat)


def plot_training_curves(history: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return
    epochs = [r["epoch"] for r in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [r["train"]["loss"] for r in history], label="train", marker="o")
    axes[0].plot(epochs, [r["val"]["loss"] for r in history], label="val", marker="o")
    axes[0].set_title("imagined-φ Huber loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, [r["train"]["corr_d_k"] for r in history], label="train", marker="o")
    axes[1].plot(epochs, [r["val"]["corr_d_k"] for r in history], label="val", marker="o")
    axes[1].set_title("corr(d, k)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _assert_on_device(module: torch.nn.Module, device: torch.device, *, name: str) -> None:
    param = next(module.parameters(), None)
    if param is None:
        return
    if param.device.type != device.type:
        raise RuntimeError(f"{name} on {param.device}, expected {device}")


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
                min_episode_len=args.k_max + HISTORY + 2,
                collector=args.collector,
            )
    finally:
        world.close()
    print(
        f"collected bank: episodes={len(bank.episodes)} steps={bank.num_steps} "
        f"collector={bank.collector}"
    )
    return bank


def run_epoch(
    model,
    reach,
    loader,
    device,
    *,
    train: bool,
    optimizer=None,
    real_frac: float,
    rng: np.random.Generator,
):
    reach.train(train)
    losses, ds, ks = [], [], []
    n_imag, n_tot = 0, 0
    non_blocking = device.type == "cuda"
    for batch in loader:
        pixels_hist = batch["pixels_hist"].to(device, non_blocking=non_blocking)
        actions_all = batch["actions_all"].to(device, non_blocking=non_blocking)
        k_int = batch["k_int"].to(device, non_blocking=non_blocking)
        pixels_t = batch["pixels_t"].to(device, non_blocking=non_blocking)
        pixels_tk = batch["pixels_tk"].to(device, non_blocking=non_blocking)
        k = batch["k"].to(device, non_blocking=non_blocking)
        B = k.size(0)

        with torch.no_grad():
            z_t = encode_frames(model, pixels_t)
            z_hist = encode_frames(model, pixels_hist)
            z_hat = imagine_futures(
                model, z_hist, actions_all, k_int, history=HISTORY
            )
            z_real_tk = encode_frames(model, pixels_tk)
            use_real = torch.tensor(
                rng.random(B) < real_frac, device=device, dtype=torch.bool
            )
            z_tk = torch.where(use_real.unsqueeze(-1), z_real_tk, z_hat)
            n_imag += int((~use_real).sum().item())
            n_tot += B

        d = reach.pairwise_distance(z_t, z_tk, detach_z=True)
        loss = F.huber_loss(d, k)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    raise RuntimeError("trunk received gradients during train_phi_imagined")
            optimizer.step()

        losses.append(float(loss.item()))
        ds.append(d.detach().cpu())
        ks.append(k.detach().cpu())

    d_cat = torch.cat(ds) if ds else torch.zeros(0)
    k_cat = torch.cat(ks) if ks else torch.zeros(0)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mean_d": float(d_cat.mean()) if d_cat.numel() else float("nan"),
        "mean_k": float(k_cat.mean()) if k_cat.numel() else float("nan"),
        "corr_d_k": pearson_corr(d_cat, k_cat) if d_cat.numel() else float("nan"),
        "frac_imagined": float(n_imag / max(n_tot, 1)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="hf_pusht")
    p.add_argument("--collector", choices=["weak", "kinematic", "goal"], default="weak")
    p.add_argument("--collect-episodes", type=int, default=256)
    p.add_argument("--collect-steps", type=int, default=24000)
    p.add_argument("--ep-horizon", type=int, default=80)
    p.add_argument("--k-max", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--samples-per-epoch", type=int, default=2048)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--real-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            if args.allow_cpu:
                device = torch.device("cpu")
            else:
                raise SystemExit("CUDA unavailable; pass --allow-cpu")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        print(f"using GPU {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"using device {device}")

    cache = Path(os.environ["STABLEWM_HOME"])
    out_dir = args.out_dir or (cache / "checkpoints" / "pusht" / "lewm_phi_imagined_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"loading frozen trunk {args.ckpt}")
    model = load_lewm_checkpoint(args.ckpt, cache_dir=cache)
    model.to(device).eval()
    model.requires_grad_(False)
    _assert_on_device(model, device, name="trunk")

    bank = collect_live_bank(args)
    transform = img_transform(224)
    train_set, val_set, split_meta = build_imagined_train_val(
        bank,
        k_max=args.k_max,
        samples_per_epoch=args.samples_per_epoch,
        val_frac=args.val_frac,
        seed=args.seed,
        img_transform=transform,
    )
    print(
        f"split train_eps={split_meta['n_train_episodes']} "
        f"val_eps={split_meta['n_val_episodes']} "
        f"real_frac={args.real_frac}"
    )

    def make_loaders(train_ds):
        tr = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_imagined,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )
        va = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_imagined,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
        )
        return tr, va

    train_loader, val_loader = make_loaders(train_set)
    reach = ReachabilityHead(distance_mode="euclidean", output_dim=64).to(device)
    optim = torch.optim.AdamW(reach.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_corr = -1.0
    best_path = out_dir / "reach.pt"
    metrics_csv = out_dir / "metrics.csv"
    curves_png = out_dir / "training_curves.png"
    if metrics_csv.exists():
        metrics_csv.unlink()

    for epoch in range(1, args.epochs + 1):
        if epoch > 1:
            train_set.pairs = sample_imagined_hindsight_pairs(
                train_set.episodes,
                k_max=args.k_max,
                n_samples=args.samples_per_epoch,
                seed=args.seed + epoch,
            )
            train_loader, val_loader = make_loaders(train_set)

        tr = run_epoch(
            model,
            reach,
            train_loader,
            device,
            train=True,
            optimizer=optim,
            real_frac=args.real_frac,
            rng=rng,
        )
        va = run_epoch(
            model,
            reach,
            val_loader,
            device,
            train=False,
            real_frac=args.real_frac,
            rng=rng,
        )
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        _append_metrics_csv(metrics_csv, row)
        plot_training_curves(history, curves_png)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss={tr['loss']:.4f} corr={tr['corr_d_k']:.3f} imag={tr['frac_imagined']:.2f}  "
            f"val_loss={va['loss']:.4f} corr={va['corr_d_k']:.3f}"
        )
        if va["corr_d_k"] == va["corr_d_k"] and va["corr_d_k"] >= best_corr:
            best_corr = va["corr_d_k"]
            torch.save(
                {
                    "reach": reach.state_dict(),
                    "meta": {
                        **row,
                        "distance_mode": "euclidean",
                        "protocol": "imagined_hindsight_H1",
                        "real_frac": args.real_frac,
                        "output_dim": 64,
                        "input_dim": 192,
                        "hidden_dim": 256,
                    },
                    "collector": args.collector,
                    "data": "live_imagined_hindsight",
                },
                best_path,
            )
            print(f"  saved {best_path} (val corr={best_corr:.3f})")

    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "history": history,
        "best_corr": best_corr,
        "split": split_meta,
        "protocol": "06_imagined_phi_H1",
    }
    (out_dir / "train_phi_imagined_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
