"""Episode- and run-level metric containers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class EpisodeRecord:
    episode_idx: int
    seed: int
    success: bool
    length: int
    truncated: bool
    steps_to_success: int | None
    min_state_distance: float | None
    final_state_distance: float | None
    min_pos_error: float | None = None
    min_angle_error: float | None = None
    min_task_score: float | None = None
    final_task_score: float | None = None
    total_contacts: int = 0
    replans: int = 0
    planning_time_s: float = 0.0


@dataclass
class RunSummary:
    success_rate: float = 0.0
    num_episodes: int = 0
    num_successes: int = 0
    mean_episode_length: float = 0.0
    mean_steps_to_success: float | None = None
    mean_min_state_distance: float | None = None
    std_min_state_distance: float | None = None
    mean_final_state_distance: float | None = None
    mean_replans: float = 0.0
    mean_planning_time_s: float = 0.0
    total_planning_time_s: float = 0.0
    wall_time_s: float = 0.0


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def summarize_episodes(episodes: list[EpisodeRecord], wall_time_s: float) -> RunSummary:
    if not episodes:
        return RunSummary(wall_time_s=wall_time_s)

    successes = [ep.success for ep in episodes]
    min_dists = [ep.min_state_distance for ep in episodes if ep.min_state_distance is not None]
    final_dists = [
        ep.final_state_distance for ep in episodes if ep.final_state_distance is not None
    ]
    steps_to_success = [
        ep.steps_to_success for ep in episodes if ep.steps_to_success is not None
    ]
    mean_steps, _ = _mean_std([float(x) for x in steps_to_success])
    mean_min, std_min = _mean_std(min_dists)
    mean_final, _ = _mean_std(final_dists)

    return RunSummary(
        success_rate=float(np.mean(successes) * 100.0),
        num_episodes=len(episodes),
        num_successes=int(sum(successes)),
        mean_episode_length=float(np.mean([ep.length for ep in episodes])),
        mean_steps_to_success=mean_steps,
        mean_min_state_distance=mean_min,
        std_min_state_distance=std_min,
        mean_final_state_distance=mean_final,
        mean_replans=float(np.mean([ep.replans for ep in episodes])),
        mean_planning_time_s=float(np.mean([ep.planning_time_s for ep in episodes])),
        total_planning_time_s=float(sum(ep.planning_time_s for ep in episodes)),
        wall_time_s=wall_time_s,
    )


def episodes_to_dict(episodes: list[EpisodeRecord]) -> list[dict[str, Any]]:
    return [asdict(ep) for ep in episodes]


def summary_to_dict(summary: RunSummary) -> dict[str, Any]:
    return asdict(summary)
