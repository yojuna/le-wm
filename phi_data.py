"""Hindsight pair sampling from live TrajectoryBank (no HF HDF5).

Pairs are *indexed* (ep, t, k). Train/val must be split by **episode** so
validation is a true held-out estimate (not a second draw from the same RNG).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from eval_logging.pairs import EpisodeTraj, TrajectoryBank


@dataclass(frozen=True)
class HindsightPair:
    """One (t, t+k) sample inside a local episode list."""

    ep_idx: int
    t: int
    k: int


def usable_episodes(
    bank: TrajectoryBank,
    *,
    k_max: int,
    require_min_length: bool = True,
) -> list[EpisodeTraj]:
    """Episodes long enough to sample k ∈ [1, k_max]."""
    out: list[EpisodeTraj] = []
    for ep in bank.episodes:
        L = len(ep)
        if require_min_length and L <= k_max:
            continue
        if L < 2:
            continue
        out.append(ep)
    if not out:
        raise RuntimeError(
            f"no episodes usable for hindsight (need length > k_max={k_max})"
        )
    return out


def split_episodes(
    episodes: list[EpisodeTraj],
    *,
    val_frac: float,
    seed: int,
) -> tuple[list[EpisodeTraj], list[EpisodeTraj]]:
    """Disjoint episode split. At least one episode on each side when possible."""
    if not episodes:
        raise ValueError("empty episode list")
    n = len(episodes)
    if n == 1:
        # Cannot hold out; duplicate warning path — train=val same ep (caller logs).
        return list(episodes), list(episodes)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    n_val = min(n_val, n - 1)  # keep ≥1 train
    val_idx = set(int(i) for i in order[:n_val])
    train = [episodes[i] for i in range(n) if i not in val_idx]
    val = [episodes[i] for i in range(n) if i in val_idx]
    return train, val


def sample_hindsight_pairs(
    episodes: list[EpisodeTraj],
    *,
    k_max: int,
    n_samples: int,
    seed: int,
) -> list[HindsightPair]:
    """Draw a fixed list of (ep_idx, t, k) with episode weighting by feasible span."""
    if not episodes:
        raise ValueError("no episodes to sample from")
    spans = np.array(
        [max(1, len(ep) - min(k_max, len(ep) - 1)) for ep in episodes],
        dtype=np.float64,
    )
    probs = spans / spans.sum()
    rng = np.random.default_rng(seed)
    pairs: list[HindsightPair] = []
    for _ in range(n_samples):
        ep_i = int(rng.choice(len(episodes), p=probs))
        L = len(episodes[ep_i])
        max_k = min(k_max, L - 1)
        k = int(rng.integers(1, max_k + 1))
        t = int(rng.integers(0, L - k))
        pairs.append(HindsightPair(ep_idx=ep_i, t=t, k=k))
    return pairs


def frame_to_tensor(frame: np.ndarray, img_transform=None) -> torch.Tensor:
    """HWC/CHW uint8|float → CHW float in [0,1], optional ImageNet transform."""
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] in (1, 3):
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
        if t.max() > 1.5:
            t = t / 255.0
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        t = torch.from_numpy(np.ascontiguousarray(arr)).float()
        if t.max() > 1.5:
            t = t / 255.0
    else:
        raise ValueError(f"unexpected frame shape {arr.shape}")
    if img_transform is not None:
        t = img_transform(t)
    return t


class LiveHindsightPairDataset(Dataset):
    """Indexed hindsight pairs over a fixed episode list.

    ``__getitem__(i)`` always returns ``pairs[i]`` (deterministic). Regenerate
    ``pairs`` between epochs if you want fresh train samples; keep val fixed.
    """

    def __init__(
        self,
        episodes: list[EpisodeTraj],
        pairs: list[HindsightPair],
        *,
        img_transform=None,
    ):
        if not episodes:
            raise ValueError("episodes must be non-empty")
        if not pairs:
            raise ValueError("pairs must be non-empty")
        self.episodes = episodes
        self.pairs = list(pairs)
        self.img_transform = img_transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        p = self.pairs[int(index)]
        ep = self.episodes[p.ep_idx]
        return {
            "pixels_t": frame_to_tensor(ep.pixels[p.t], self.img_transform),
            "pixels_tk": frame_to_tensor(ep.pixels[p.t + p.k], self.img_transform),
            "k": torch.tensor(float(p.k), dtype=torch.float32),
        }


def build_train_val_datasets(
    bank: TrajectoryBank,
    *,
    k_max: int,
    samples_per_epoch: int,
    val_frac: float,
    seed: int,
    img_transform=None,
    val_samples: int | None = None,
) -> tuple[LiveHindsightPairDataset, LiveHindsightPairDataset, dict]:
    """Episode-held-out train/val datasets with fixed pair lists."""
    episodes = usable_episodes(bank, k_max=k_max)
    train_eps, val_eps = split_episodes(episodes, val_frac=val_frac, seed=seed)
    n_val_pairs = int(val_samples or max(64, int(samples_per_epoch * val_frac)))
    train_pairs = sample_hindsight_pairs(
        train_eps, k_max=k_max, n_samples=samples_per_epoch, seed=seed
    )
    val_pairs = sample_hindsight_pairs(
        val_eps, k_max=k_max, n_samples=n_val_pairs, seed=seed + 1
    )
    train_ds = LiveHindsightPairDataset(
        train_eps, train_pairs, img_transform=img_transform
    )
    val_ds = LiveHindsightPairDataset(val_eps, val_pairs, img_transform=img_transform)
    meta = {
        "n_usable_episodes": len(episodes),
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "same_episode_fallback": len(episodes) == 1,
        "train_val_episode_overlap": bool(
            set(id(e) for e in train_eps) & set(id(e) for e in val_eps)
        ),
    }
    return train_ds, val_ds, meta


def collate_hindsight(batch):
    return {
        "pixels_t": torch.stack([b["pixels_t"] for b in batch], dim=0),
        "pixels_tk": torch.stack([b["pixels_tk"] for b in batch], dim=0),
        "k": torch.stack([b["k"] for b in batch], dim=0),
    }


# Back-compat alias
HindsightPairDataset = LiveHindsightPairDataset
