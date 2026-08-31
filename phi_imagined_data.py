"""Hindsight pairs with history frames + actions for imagined-future φ training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from eval_logging.pairs import EpisodeTraj, TrajectoryBank
from phi_data import (
    HindsightPair,
    frame_to_tensor,
    sample_hindsight_pairs,
    split_episodes,
    usable_episodes,
)

HISTORY = 3
ACTION_DIM = 10


def pad_action(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[: min(len(a), ACTION_DIM)] = a[:ACTION_DIM]
    return out


def sample_imagined_hindsight_pairs(
    episodes: list[EpisodeTraj],
    *,
    k_max: int,
    n_samples: int,
    seed: int,
    history: int = HISTORY,
) -> list[HindsightPair]:
    """Like sample_hindsight_pairs but require t >= history-1 for predictor warmup."""
    if not episodes:
        raise ValueError("no episodes")
    spans = []
    for ep in episodes:
        L = len(ep.pixels) if ep.pixels else len(ep)
        # t in [history-1, L-2], k in [1, min(k_max, L-1-t)]
        max_t = L - 2
        min_t = history - 1
        span = max(0, max_t - min_t + 1)
        spans.append(float(max(span, 1)))
    spans_arr = np.asarray(spans, dtype=np.float64)
    probs = spans_arr / spans_arr.sum()
    rng = np.random.default_rng(seed)
    pairs: list[HindsightPair] = []
    for _ in range(n_samples):
        ep_i = int(rng.choice(len(episodes), p=probs))
        ep = episodes[ep_i]
        L = len(ep.pixels) if ep.pixels else len(ep)
        min_t = history - 1
        max_t = L - 2
        if max_t < min_t:
            # fallback: skip impossible ep by resampling
            continue
        t = int(rng.integers(min_t, max_t + 1))
        max_k = min(k_max, L - 1 - t)
        if max_k < 1:
            continue
        k = int(rng.integers(1, max_k + 1))
        pairs.append(HindsightPair(ep_idx=ep_i, t=t, k=k))
    if len(pairs) < n_samples:
        # fill by retrying
        while len(pairs) < n_samples:
            more = sample_imagined_hindsight_pairs(
                episodes,
                k_max=k_max,
                n_samples=n_samples - len(pairs),
                seed=seed + len(pairs) + 1,
                history=history,
            )
            pairs.extend(more)
            if not more:
                break
    return pairs[:n_samples]


class LiveImaginedHindsightDataset(Dataset):
    """Returns history pixels/actions + start/goal pixels for imagined training."""

    def __init__(
        self,
        episodes: list[EpisodeTraj],
        pairs: list[HindsightPair],
        *,
        img_transform=None,
        history: int = HISTORY,
    ):
        self.episodes = episodes
        self.pairs = list(pairs)
        self.img_transform = img_transform
        self.history = int(history)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        p = self.pairs[int(index)]
        ep = self.episodes[p.ep_idx]
        t, k = p.t, p.k
        HS = self.history

        hist_pix = []
        acts_all = []
        # actions for frames [t-HS+1, t+k-1] — length HS+k-1? need HS+k for k appends after hist
        # hist actions: [t-HS+1, t]; then appends a_{t+1}..a_{t+k-1} wait
        # Loop k times appending acts[HS], acts[HS+1], ... → need acts length HS+k
        # Frame indices for actions: (t-HS+1) .. (t-HS+1 + HS+k - 1) = (t-HS+1)..(t+k)
        # Last action a_{t+k} unused for final pred but pad OK
        for i in range(t - HS + 1, t + 1):
            hist_pix.append(frame_to_tensor(ep.pixels[i], self.img_transform))
        for i in range(t - HS + 1, t + k + 1):
            if 0 <= i < len(ep.action):
                acts_all.append(pad_action(ep.action[i]))
            else:
                acts_all.append(np.zeros(ACTION_DIM, np.float32))

        return {
            "pixels_hist": torch.stack(hist_pix, dim=0),  # (HS,C,H,W)
            "actions_all": torch.from_numpy(np.stack(acts_all, axis=0)),  # (HS+k, 10)
            "pixels_t": frame_to_tensor(ep.pixels[t], self.img_transform),
            "pixels_tk": frame_to_tensor(ep.pixels[t + k], self.img_transform),
            "k": torch.tensor(float(k), dtype=torch.float32),
        }


def collate_imagined(batch):
    # actions_all length = HS + k (variable k) — pad to max
    HS = batch[0]["pixels_hist"].size(0)
    max_k = max(int(b["k"].item()) for b in batch)
    B = len(batch)
    acts = torch.zeros(B, HS + max_k, ACTION_DIM, dtype=torch.float32)
    k_int = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        kk = int(b["k"].item())
        k_int[i] = kk
        seq = b["actions_all"]
        acts[i, : seq.size(0)] = seq[: HS + kk]
    return {
        "pixels_hist": torch.stack([b["pixels_hist"] for b in batch], dim=0),
        "actions_all": acts,
        "k_int": k_int,
        "pixels_t": torch.stack([b["pixels_t"] for b in batch], dim=0),
        "pixels_tk": torch.stack([b["pixels_tk"] for b in batch], dim=0),
        "k": torch.stack([b["k"] for b in batch], dim=0),
    }


def build_imagined_train_val(
    bank: TrajectoryBank,
    *,
    k_max: int,
    samples_per_epoch: int,
    val_frac: float,
    seed: int,
    img_transform=None,
    val_samples: int | None = None,
    history: int = HISTORY,
):
    episodes = usable_episodes(bank, k_max=k_max)
    # also need length > history
    episodes = [
        e
        for e in episodes
        if (len(e.pixels) if e.pixels else len(e)) > history + 1
    ]
    if not episodes:
        raise RuntimeError("no episodes long enough for imagined hindsight")
    train_eps, val_eps = split_episodes(episodes, val_frac=val_frac, seed=seed)
    n_val = int(val_samples or max(64, int(samples_per_epoch * val_frac)))
    train_pairs = sample_imagined_hindsight_pairs(
        train_eps, k_max=k_max, n_samples=samples_per_epoch, seed=seed, history=history
    )
    val_pairs = sample_imagined_hindsight_pairs(
        val_eps, k_max=k_max, n_samples=n_val, seed=seed + 1, history=history
    )
    train_ds = LiveImaginedHindsightDataset(
        train_eps, train_pairs, img_transform=img_transform, history=history
    )
    val_ds = LiveImaginedHindsightDataset(
        val_eps, val_pairs, img_transform=img_transform, history=history
    )
    meta = {
        "n_train_episodes": len(train_eps),
        "n_val_episodes": len(val_eps),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "history": history,
        "same_episode_fallback": len(episodes) == 1,
    }
    return train_ds, val_ds, meta
