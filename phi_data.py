"""Episode-aware hindsight pair sampling for training φ."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EpisodeIndex:
    """Maps global row indices for one episode."""

    episode_id: int
    row_indices: np.ndarray  # indices into the flat dataset
    length: int


def build_episode_index(dataset) -> list[EpisodeIndex]:
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep = np.asarray(dataset.get_col_data(col))
    step = np.asarray(dataset.get_col_data("step_idx"))
    episodes: list[EpisodeIndex] = []
    for ep_id in np.unique(ep):
        mask = ep == ep_id
        rows = np.nonzero(mask)[0]
        # sort by step within episode
        order = np.argsort(step[rows])
        rows = rows[order]
        episodes.append(
            EpisodeIndex(episode_id=int(ep_id), row_indices=rows, length=len(rows))
        )
    return episodes


class HindsightPairDataset(Dataset):
    """Sample (pixels_t, pixels_{t+k}, k) with k in [1, k_max]."""

    def __init__(
        self,
        dataset,
        *,
        k_max: int = 25,
        img_transform=None,
        episodes: list[EpisodeIndex] | None = None,
        samples_per_epoch: int | None = None,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.k_max = int(k_max)
        self.img_transform = img_transform
        self.episodes = episodes or build_episode_index(dataset)
        self.episodes = [e for e in self.episodes if e.length > self.k_max]
        if not self.episodes:
            raise RuntimeError(
                f"no episodes longer than k_max={self.k_max}; check dataset"
            )
        self.samples_per_epoch = int(
            samples_per_epoch or sum(e.length for e in self.episodes)
        )
        self.rng = np.random.default_rng(seed)
        # precompute cumulative lengths for weighted sampling by episode size
        self._lengths = np.array([e.length - self.k_max for e in self.episodes])
        self._cum = np.cumsum(self._lengths)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_episode_and_t(self):
        r = self.rng.integers(0, int(self._cum[-1]))
        ep_i = int(np.searchsorted(self._cum, r, side="right"))
        ep = self.episodes[ep_i]
        max_t = ep.length - self.k_max - 1
        t = int(self.rng.integers(0, max_t + 1))
        k = int(self.rng.integers(1, self.k_max + 1))
        return ep, t, k

    def _load_pixels(self, row_idx: int) -> torch.Tensor:
        row = self.dataset.get_row_data(int(row_idx))
        pixels = row["pixels"]
        if isinstance(pixels, np.ndarray):
            pixels = torch.from_numpy(pixels)
        if self.img_transform is not None:
            pixels = self.img_transform(pixels)
        else:
            pixels = pixels.float()
            if pixels.max() > 1.5:
                pixels = pixels / 255.0
        return pixels

    def __getitem__(self, index: int):
        # index only used for Dataset protocol; sampling is random each call
        ep, t, k = self._sample_episode_and_t()
        row_t = int(ep.row_indices[t])
        row_tk = int(ep.row_indices[t + k])
        pix_t = self._load_pixels(row_t)
        pix_tk = self._load_pixels(row_tk)
        return {
            "pixels_t": pix_t,
            "pixels_tk": pix_tk,
            "k": torch.tensor(k, dtype=torch.float32),
        }


def collate_hindsight(batch):
    return {
        "pixels_t": torch.stack([b["pixels_t"] for b in batch], dim=0),
        "pixels_tk": torch.stack([b["pixels_tk"] for b in batch], dim=0),
        "k": torch.stack([b["k"] for b in batch], dim=0),
    }
