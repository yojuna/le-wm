"""IQL transition+goal sampling for Protocol T3 (Destrade Eq. 1).

Each sample: (pixels_t, pixels_tp1, pixels_g, not_at_goal) with
goals = trajectory terminal OR random bank frame (configurable mix).
Equality 1_{s≠g} is exact on (episode_idx, timestep), not pose tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from eval_logging.pairs import EpisodeTraj, TrajectoryBank
from phi_data import frame_to_tensor, split_episodes


@dataclass(frozen=True)
class IQLTransition:
    """Indexed (s_t, s_{t+1}, g) inside a local episode list."""

    ep_idx: int
    t: int
    g_ep_idx: int
    g_t: int

    @property
    def not_at_goal(self) -> bool:
        return (self.ep_idx, self.t) != (self.g_ep_idx, self.g_t)


def episode_frame_len(ep: EpisodeTraj) -> int:
    """Prefer pixels length (what we index); EpisodeTraj.__len__ is len(state)."""
    if ep.pixels:
        return len(ep.pixels)
    return len(ep)


def usable_iql_episodes(
    bank: TrajectoryBank,
    *,
    require_min_length: bool = True,
) -> list[EpisodeTraj]:
    """Episodes with at least one transition (frame len >= 2)."""
    out: list[EpisodeTraj] = []
    for ep in bank.episodes:
        if require_min_length and episode_frame_len(ep) < 2:
            continue
        if episode_frame_len(ep) < 2:
            continue
        out.append(ep)
    if not out:
        raise RuntimeError("no episodes usable for IQL (need length >= 2)")
    return out


def sample_iql_transitions(
    episodes: list[EpisodeTraj],
    *,
    n_samples: int,
    seed: int,
    terminal_goal_frac: float = 0.5,
) -> list[IQLTransition]:
    """Draw fixed (ep,t,g) list. Goals: terminal vs random bank frame."""
    if not episodes:
        raise ValueError("no episodes to sample from")
    if not (0.0 <= terminal_goal_frac <= 1.0):
        raise ValueError(f"terminal_goal_frac must be in [0,1], got {terminal_goal_frac}")

    lengths = np.array([episode_frame_len(ep) for ep in episodes], dtype=np.int64)
    # Weight by number of valid t ∈ [0, L-2]
    spans = np.maximum(lengths - 1, 1).astype(np.float64)
    probs = spans / spans.sum()
    rng = np.random.default_rng(seed)

    # Flat index of all frames for random goals
    frame_eps: list[int] = []
    frame_ts: list[int] = []
    for ei, ep in enumerate(episodes):
        for ti in range(episode_frame_len(ep)):
            frame_eps.append(ei)
            frame_ts.append(ti)
    frame_eps_arr = np.asarray(frame_eps, dtype=np.int64)
    frame_ts_arr = np.asarray(frame_ts, dtype=np.int64)

    out: list[IQLTransition] = []
    for _ in range(n_samples):
        ep_i = int(rng.choice(len(episodes), p=probs))
        L = int(lengths[ep_i])
        t = int(rng.integers(0, L - 1))
        if rng.random() < terminal_goal_frac:
            g_ep, g_t = ep_i, L - 1
        else:
            fi = int(rng.integers(0, len(frame_eps_arr)))
            g_ep = int(frame_eps_arr[fi])
            g_t = int(frame_ts_arr[fi])
        out.append(IQLTransition(ep_idx=ep_i, t=t, g_ep_idx=g_ep, g_t=g_t))
    return out


class LiveIQLTransitionDataset(Dataset):
    """Indexed IQL transitions over a fixed episode list."""

    def __init__(
        self,
        episodes: list[EpisodeTraj],
        transitions: list[IQLTransition],
        *,
        img_transform=None,
    ):
        if not episodes:
            raise ValueError("episodes must be non-empty")
        if not transitions:
            raise ValueError("transitions must be non-empty")
        self.episodes = episodes
        self.transitions = list(transitions)
        self.img_transform = img_transform

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, index: int):
        tr = self.transitions[int(index)]
        ep = self.episodes[tr.ep_idx]
        g_ep = self.episodes[tr.g_ep_idx]
        return {
            "pixels_t": frame_to_tensor(ep.pixels[tr.t], self.img_transform),
            "pixels_tp1": frame_to_tensor(ep.pixels[tr.t + 1], self.img_transform),
            "pixels_g": frame_to_tensor(g_ep.pixels[tr.g_t], self.img_transform),
            "not_at_goal": torch.tensor(
                1.0 if tr.not_at_goal else 0.0, dtype=torch.float32
            ),
        }


def build_iql_train_val_datasets(
    bank: TrajectoryBank,
    *,
    samples_per_epoch: int,
    val_frac: float,
    seed: int,
    img_transform=None,
    val_samples: int | None = None,
    terminal_goal_frac: float = 0.5,
) -> tuple[LiveIQLTransitionDataset, LiveIQLTransitionDataset, dict]:
    """Episode-held-out train/val IQL datasets with fixed transition lists."""
    episodes = usable_iql_episodes(bank)
    train_eps, val_eps = split_episodes(episodes, val_frac=val_frac, seed=seed)
    n_val = int(val_samples or max(64, int(samples_per_epoch * val_frac)))
    train_tr = sample_iql_transitions(
        train_eps,
        n_samples=samples_per_epoch,
        seed=seed,
        terminal_goal_frac=terminal_goal_frac,
    )
    val_tr = sample_iql_transitions(
        val_eps,
        n_samples=n_val,
        seed=seed + 1,
        terminal_goal_frac=terminal_goal_frac,
    )
    train_ds = LiveIQLTransitionDataset(
        train_eps, train_tr, img_transform=img_transform
    )
    val_ds = LiveIQLTransitionDataset(val_eps, val_tr, img_transform=img_transform)
    frac_eq = float(np.mean([0.0 if t.not_at_goal else 1.0 for t in train_tr]))
    meta = {
        "n_usable_episodes": len(episodes),
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_train_transitions": len(train_tr),
        "n_val_transitions": len(val_tr),
        "terminal_goal_frac": terminal_goal_frac,
        "train_frac_s_eq_g": frac_eq,
        "same_episode_fallback": len(episodes) == 1,
        "train_val_episode_overlap": bool(
            set(id(e) for e in train_eps) & set(id(e) for e in val_eps)
        ),
    }
    return train_ds, val_ds, meta


def collate_iql(batch):
    return {
        "pixels_t": torch.stack([b["pixels_t"] for b in batch], dim=0),
        "pixels_tp1": torch.stack([b["pixels_tp1"] for b in batch], dim=0),
        "pixels_g": torch.stack([b["pixels_g"] for b in batch], dim=0),
        "not_at_goal": torch.stack([b["not_at_goal"] for b in batch], dim=0),
    }
