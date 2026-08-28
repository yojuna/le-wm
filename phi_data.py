"""Hindsight pair sampling from live TrajectoryBank (no HF HDF5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from eval_logging.pairs import EpisodeTraj, TrajectoryBank


@dataclass
class _EpView:
    ep: EpisodeTraj
    length: int


class LiveHindsightPairDataset(Dataset):
    """Sample (pixels_t, pixels_{t+k}, k) from simulator-collected episodes."""

    def __init__(
        self,
        bank: TrajectoryBank,
        *,
        k_max: int = 25,
        img_transform=None,
        samples_per_epoch: int | None = None,
        seed: int = 0,
        require_min_length: bool = True,
    ):
        self.k_max = int(k_max)
        self.img_transform = img_transform
        self.rng = np.random.default_rng(seed)

        views: list[_EpView] = []
        for ep in bank.episodes:
            L = len(ep)
            if require_min_length and L <= self.k_max:
                continue
            if L < 2:
                continue
            views.append(_EpView(ep=ep, length=L))
        if not views:
            raise RuntimeError(
                f"no episodes usable for hindsight (need length > k_max={self.k_max})"
            )
        self.episodes = views
        self._span = np.array(
            [max(1, v.length - min(self.k_max, v.length - 1)) for v in self.episodes],
            dtype=np.int64,
        )
        self._cum = np.cumsum(self._span)
        self.samples_per_epoch = int(
            samples_per_epoch or int(self._cum[-1]) * 4
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_ep_t_k(self) -> tuple[EpisodeTraj, int, int]:
        r = int(self.rng.integers(0, int(self._cum[-1])))
        ep_i = int(np.searchsorted(self._cum, r, side="right"))
        view = self.episodes[ep_i]
        max_k = min(self.k_max, view.length - 1)
        k = int(self.rng.integers(1, max_k + 1))
        t = int(self.rng.integers(0, view.length - k))
        return view.ep, t, k

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        pixels = torch.as_tensor(np.asarray(frame))
        if self.img_transform is not None:
            return self.img_transform(pixels)
        pixels = pixels.float()
        if pixels.max() > 1.5:
            pixels = pixels / 255.0
        # HWC -> CHW if needed
        if pixels.ndim == 3 and pixels.shape[-1] in (1, 3):
            pixels = pixels.permute(2, 0, 1)
        return pixels

    def __getitem__(self, index: int):
        ep, t, k = self._sample_ep_t_k()
        return {
            "pixels_t": self._to_tensor(ep.pixels[t]),
            "pixels_tk": self._to_tensor(ep.pixels[t + k]),
            "k": torch.tensor(float(k), dtype=torch.float32),
        }


def collate_hindsight(batch):
    return {
        "pixels_t": torch.stack([b["pixels_t"] for b in batch], dim=0),
        "pixels_tk": torch.stack([b["pixels_tk"] for b in batch], dim=0),
        "k": torch.stack([b["k"] for b in batch], dim=0),
    }


# Back-compat alias used by older notes
HindsightPairDataset = LiveHindsightPairDataset
