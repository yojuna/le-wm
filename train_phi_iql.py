#!/usr/bin/env python3
"""Train φ with Destrade Eq. (1) quasimetric IQL on a frozen LeWM trunk.

Protocol T3 — see docs/03_quasimetric_iql_t3.md.
Defaults match paper VF_quasi: γ=0.93, τ=0.60, IQE-sum, kinematic bank.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
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
from iql_loss import iql_vf_loss
from phi_iql_data import (
    build_iql_train_val_datasets,
    collate_iql,
    sample_iql_transitions,
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
    info = {"pixels": pixels.unsqueeze(1)}
    out = model.encode(info)
    return out["emb"][:, 0]


def _append_metrics_csv(path: Path, row: dict) -> None:
    import csv

    flat = {
        "epoch": row["epoch"],
        "train_loss": row["train"]["loss"],
        "train_mean_d": row["train"]["mean_d"],
        "train_mean_V": row["train"]["mean_V"],
        "train_frac_eq": row["train"]["frac_s_eq_g"],
        "train_mean_td": row["train"]["mean_td"],
        "val_loss": row["val"]["loss"],
        "val_mean_d": row["val"]["mean_d"],
        "val_mean_V": row["val"]["mean_V"],
        "val_frac_eq": row["val"]["frac_s_eq_g"],
        "val_mean_td": row["val"]["mean_td"],
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
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("L_VF")
    axes[0].set_title("IQL expectile loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [r["train"]["mean_d"] for r in history], label="train d", marker="o")
    axes[1].plot(epochs, [r["val"]["mean_d"] for r in history], label="val d", marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("mean IQE-sum d")
    axes[1].set_title("planning distance")
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
        raise RuntimeError(
            f"{name} is on {param.device}, expected {device}. "
            "Training would not use the requested GPU."
        )


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
                min_episode_len=3,
                collector=args.collector,
            )
    finally:
        world.close()

    usable = sum(1 for ep in bank.episodes if len(ep.pixels) >= 2)
    print(
        f"collected bank: episodes={len(bank.episodes)} steps={bank.num_steps} "
        f"usable(>=2)={usable} collector={bank.collector} "
        f"success_eps={bank.num_success_episodes}"
    )
    if usable == 0:
        raise RuntimeError(
            "no episodes with length>=2; increase --collect-episodes/--collect-steps"
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
    gamma: float,
    tau: float,
):
    reach.train(train)
    losses, ds, vs, tds, eqs = [], [], [], [], []
    non_blocking = isinstance(device, torch.device) and device.type == "cuda"
    for batch in loader:
        pixels_t = batch["pixels_t"].to(device, non_blocking=non_blocking)
        pixels_tp1 = batch["pixels_tp1"].to(device, non_blocking=non_blocking)
        pixels_g = batch["pixels_g"].to(device, non_blocking=non_blocking)
        not_at_goal = batch["not_at_goal"].to(device, non_blocking=non_blocking)

        with torch.no_grad():
            z_t = encode_frames(model, pixels_t)
            z_tp1 = encode_frames(model, pixels_tp1)
            z_g = encode_frames(model, pixels_g)

        V_t = reach.value(z_t, z_g, detach_z=True)
        with torch.no_grad():
            V_tp1 = reach.value(z_tp1, z_g, detach_z=True)
        d = -V_t.detach()
        td = (-not_at_goal + gamma * V_tp1 - V_t).detach()

        loss = iql_vf_loss(V_t, V_tp1, not_at_goal, gamma=gamma, tau=tau)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None:
                    raise RuntimeError("trunk received gradients during train_phi_iql")
            optimizer.step()

        losses.append(float(loss.item()))
        ds.append(d.cpu())
        vs.append(V_t.detach().cpu())
        tds.append(td.cpu())
        eqs.append((1.0 - not_at_goal).detach().cpu())

    d_cat = torch.cat(ds) if ds else torch.zeros(0)
    v_cat = torch.cat(vs) if vs else torch.zeros(0)
    td_cat = torch.cat(tds) if tds else torch.zeros(0)
    eq_cat = torch.cat(eqs) if eqs else torch.zeros(0)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "mean_d": float(d_cat.mean()) if d_cat.numel() else float("nan"),
        "mean_V": float(v_cat.mean()) if v_cat.numel() else float("nan"),
        "mean_td": float(td_cat.mean()) if td_cat.numel() else float("nan"),
        "frac_s_eq_g": float(eq_cat.mean()) if eq_cat.numel() else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default="hf_pusht")
    p.add_argument(
        "--collector",
        choices=["weak", "kinematic", "goal"],
        default="kinematic",
        help="live rollout policy (default: kinematic; Destrade warns weak hurts IQL)",
    )
    p.add_argument("--collect-episodes", type=int, default=256)
    p.add_argument("--collect-steps", type=int, default=24000)
    p.add_argument("--ep-horizon", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--samples-per-epoch", type=int, default=4096)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--terminal-goal-frac", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.93, help="VF_quasi discount")
    p.add_argument("--tau", type=float, default=0.60, help="VF_quasi expectile")
    p.add_argument("--phi-dim", type=int, default=64)
    p.add_argument("--iqe-k", type=int, default=8)
    p.add_argument("--iqe-l", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    if args.phi_dim != args.iqe_k * args.iqe_l:
        raise SystemExit(
            f"--phi-dim={args.phi_dim} must equal iqe_k*iqe_l="
            f"{args.iqe_k * args.iqe_l}"
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            if args.allow_cpu:
                print("WARNING: CUDA unavailable; falling back to CPU (--allow-cpu)")
                device = torch.device("cpu")
            else:
                raise SystemExit(
                    "CUDA requested but torch.cuda.is_available() is False. "
                    "Fix GPU/PyTorch, or pass --allow-cpu."
                )
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
    out_dir = args.out_dir or (cache / "checkpoints" / "pusht" / "lewm_phi_iql_v1")
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
    train_set, val_set, split_meta = build_iql_train_val_datasets(
        bank,
        samples_per_epoch=args.samples_per_epoch,
        val_frac=args.val_frac,
        seed=args.seed,
        img_transform=transform,
        terminal_goal_frac=args.terminal_goal_frac,
    )
    print(
        f"episode split: train_eps={split_meta['n_train_episodes']} "
        f"val_eps={split_meta['n_val_episodes']} "
        f"train_n={split_meta['n_train_transitions']} "
        f"val_n={split_meta['n_val_transitions']} "
        f"train_frac_s==g={split_meta['train_frac_s_eq_g']:.4f}"
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
            collate_fn=collate_iql,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
        )
        va = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_iql,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
        )
        return tr, va

    train_loader, val_loader = make_loaders(train_set)

    reach = ReachabilityHead(
        input_dim=192,
        output_dim=args.phi_dim,
        distance_mode="iqe_sum",
        iqe_k=args.iqe_k,
        iqe_l=args.iqe_l,
    ).to(device)
    _assert_on_device(reach, device, name="reach")
    optim = torch.optim.AdamW(reach.parameters(), lr=args.lr, weight_decay=1e-4)

    history = []
    best_val = float("inf")
    best_path = out_dir / "reach.pt"
    metrics_csv = out_dir / "metrics.csv"
    curves_png = out_dir / "training_curves.png"
    if metrics_csv.exists():
        metrics_csv.unlink()

    head_meta = {
        "distance_mode": "iqe_sum",
        "iqe_k": args.iqe_k,
        "iqe_l": args.iqe_l,
        "output_dim": args.phi_dim,
        "input_dim": 192,
        "hidden_dim": 256,
        "gamma": args.gamma,
        "tau": args.tau,
        "protocol": "T3_quasimetric_iql",
    }

    for epoch in range(1, args.epochs + 1):
        if epoch > 1:
            train_set.transitions = sample_iql_transitions(
                train_set.episodes,
                n_samples=args.samples_per_epoch,
                seed=args.seed + epoch,
                terminal_goal_frac=args.terminal_goal_frac,
            )
            train_loader, val_loader = make_loaders(train_set)

        tr = run_epoch(
            model,
            reach,
            train_loader,
            device,
            train=True,
            optimizer=optim,
            gamma=args.gamma,
            tau=args.tau,
        )
        va = run_epoch(
            model,
            reach,
            val_loader,
            device,
            train=False,
            gamma=args.gamma,
            tau=args.tau,
        )
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        _append_metrics_csv(metrics_csv, row)
        plot_training_curves(history, curves_png)
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_loss={tr['loss']:.4f} d={tr['mean_d']:.3f} V={tr['mean_V']:.3f} "
            f"eq={tr['frac_s_eq_g']:.3f}  "
            f"val_loss={va['loss']:.4f} d={va['mean_d']:.3f}  "
            f"[wrote {curves_png.name}, {metrics_csv.name}]"
        )
        if va["loss"] == va["loss"] and va["loss"] <= best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "reach": reach.state_dict(),
                    "meta": {**head_meta, **row},
                    "collector": args.collector,
                    "data": "live_trajectory_bank",
                },
                best_path,
            )
            print(f"  saved {best_path} (val loss={best_val:.4f})")

    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "history": history,
        "best_val_loss": best_val,
        "bank_episodes": len(bank.episodes),
        "bank_steps": bank.num_steps,
        "data_source": "live_simulator",
        "split": split_meta,
        "head": head_meta,
        "metrics_csv": str(metrics_csv),
        "training_curves": str(curves_png),
    }
    meta_path = out_dir / "train_phi_iql_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}")
    print(f"metrics CSV: {metrics_csv}")
    print(f"curves plot: {curves_png}")


if __name__ == "__main__":
    main()
