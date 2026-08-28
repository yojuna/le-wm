"""Collect per-step and per-episode metrics during World rollouts."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from eval_logging.collector import (
    EpisodeRecord,
    RunSummary,
    episodes_to_dict,
    summarize_episodes,
    summary_to_dict,
)
from eval_logging.config import EvalLogConfig
from eval_logging.extractors import extract_step_metrics


@dataclass
class _ActiveEpisode:
    length: int = 0
    truncated: bool = False
    steps_to_success: int | None = None
    min_state_distance: float | None = None
    final_state_distance: float | None = None
    min_pos_error: float | None = None
    min_angle_error: float | None = None
    min_task_score: float | None = None
    final_task_score: float | None = None
    total_contacts: int = 0
    replans: int = 0
    planning_time_s: float = 0.0


class EvalRunLogger:
    """Track robotics planning metrics and write experiment artifacts."""

    def __init__(
        self,
        *,
        env_key: str,
        config: EvalLogConfig,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.env_key = env_key
        self.config = config
        self.meta = meta or {}
        self._active: dict[int, _ActiveEpisode] = {}
        self.episodes: list[EpisodeRecord] = []
        self._run_started: float | None = None
        self.total_replans: int = 0
        self.total_planning_time_s: float = 0.0

    def begin_run(self) -> None:
        self._run_started = time.perf_counter()

    def note_planning(self, duration_s: float) -> None:
        if not self.config.log_planning:
            return
        self.total_replans += 1
        self.total_planning_time_s += duration_s
        if not self._active:
            self._active[0] = _ActiveEpisode()
        for ep in self._active.values():
            ep.replans += 1
            ep.planning_time_s += duration_s

    def on_step(self, world, env_idx: int) -> None:
        if env_idx not in self._active:
            self._active[env_idx] = _ActiveEpisode()

        ep = self._active[env_idx]
        ep.length += 1

        metrics = extract_step_metrics(self.env_key, world, env_idx)
        if metrics.state_distance is not None:
            ep.final_state_distance = metrics.state_distance
            ep.min_state_distance = (
                metrics.state_distance
                if ep.min_state_distance is None
                else min(ep.min_state_distance, metrics.state_distance)
            )
        if metrics.pos_error is not None:
            ep.min_pos_error = (
                metrics.pos_error
                if ep.min_pos_error is None
                else min(ep.min_pos_error, metrics.pos_error)
            )
        if metrics.angle_error is not None:
            ep.min_angle_error = (
                metrics.angle_error
                if ep.min_angle_error is None
                else min(ep.min_angle_error, metrics.angle_error)
            )
        if metrics.task_score is not None:
            ep.final_task_score = metrics.task_score
            ep.min_task_score = (
                metrics.task_score
                if ep.min_task_score is None
                else min(ep.min_task_score, metrics.task_score)
            )
        if metrics.n_contacts is not None:
            ep.total_contacts += metrics.n_contacts

        success = bool(world.terminateds[env_idx])
        if success and ep.steps_to_success is None:
            ep.steps_to_success = ep.length

        if bool(world.truncateds[env_idx]):
            ep.truncated = True

    def on_episode_done(self, env_idx: int, episode_idx: int, world) -> EpisodeRecord:
        if env_idx not in self._active:
            self._active[env_idx] = _ActiveEpisode()

        ep = self._active.pop(env_idx)
        if bool(world.truncateds[env_idx]):
            ep.truncated = True

        success = bool(world.terminateds[env_idx])
        record = EpisodeRecord(
            episode_idx=episode_idx,
            seed=int(world.envs.seeds[env_idx]),
            success=success,
            length=ep.length,
            truncated=ep.truncated,
            steps_to_success=ep.steps_to_success,
            min_state_distance=ep.min_state_distance,
            final_state_distance=ep.final_state_distance,
            min_pos_error=ep.min_pos_error,
            min_angle_error=ep.min_angle_error,
            min_task_score=ep.min_task_score,
            final_task_score=ep.final_task_score,
            total_contacts=ep.total_contacts,
            replans=ep.replans,
            planning_time_s=ep.planning_time_s,
        )
        self.episodes.append(record)

        if self.config.log_episode_events and not self.config.quiet:
            self._print_episode(record)

        return record

    def finalize(self, *, interrupted: bool = False) -> dict[str, Any]:
        wall_time = 0.0
        if self._run_started is not None:
            wall_time = time.perf_counter() - self._run_started

        summary = summarize_episodes(self.episodes, wall_time)
        payload = self._build_payload(summary, interrupted=interrupted)

        if self.config.save_json:
            path = self._write_json(payload)
            payload["output_json"] = str(path)
        if self.config.save_episode_csv:
            path = self._write_csv()
            payload["output_csv"] = str(path)

        if not self.config.quiet:
            self._print_summary(summary, payload)

        return payload

    def _build_payload(self, summary: RunSummary, *, interrupted: bool = False) -> dict[str, Any]:
        return {
            "meta": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "env": self.env_key,
                "protocol": "live_reset",
                "interrupted": interrupted,
                "completed_episodes": len(self.episodes),
                "metric_notes": {
                    "success_rate": (
                        "Fraction of episodes with env terminated=True "
                        "(PushT: pose tolerance vs goal_state; Reacher: qpos_match). "
                        "For online_goal_offset, success is OR'd over the eval budget "
                        "like World._evaluate_from_dataset."
                    ),
                    "min_state_distance": (
                        "Best (lowest) L2 distance to goal state during the episode. "
                        "Includes velocities on PushT — prefer min_pos_error."
                    ),
                    "steps_to_success": (
                        "Env step index at first success; None if the episode failed."
                    ),
                },
                **self.meta,
            },
            "aggregate": {
                **summary_to_dict(summary),
                "total_replans": self.total_replans,
                "total_planning_time_s": self.total_planning_time_s,
            },
            "episodes": episodes_to_dict(self.episodes),
            "success_rate": summary.success_rate,
            "episode_successes": np.array([ep.success for ep in self.episodes]),
            "seeds": np.array([ep.seed for ep in self.episodes], dtype=np.int64),
        }

    def _episode_log_dir(self) -> Path:
        run_name = self.config.resolve_run_name(
            self.env_key, int(self.meta.get("seed", 0))
        )
        out = self.config.output_dir / self.env_key / run_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _write_json(self, payload: dict[str, Any]) -> Path:
        out_dir = self._episode_log_dir()
        path = out_dir / "metrics.json"

        serializable = json.loads(
            json.dumps(
                payload,
                default=self._json_default,
            )
        )
        path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")
        return path

    def _write_csv(self) -> Path:
        out_dir = self._episode_log_dir()
        path = out_dir / "episodes.csv"
        if not self.episodes:
            path.write_text("")
            return path

        rows = episodes_to_dict(self.episodes)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")

    def _print_episode(self, record: EpisodeRecord) -> None:
        parts = [
            f"episode {record.episode_idx + 1}",
            f"seed={record.seed}",
            f"success={record.success}",
            f"length={record.length}",
        ]
        if record.steps_to_success is not None:
            parts.append(f"steps_to_success={record.steps_to_success}")
        if record.min_state_distance is not None:
            parts.append(f"min_dist={record.min_state_distance:.2f}")
        if record.final_state_distance is not None:
            parts.append(f"final_dist={record.final_state_distance:.2f}")
        if self.env_key == "pusht" and record.min_pos_error is not None:
            parts.append(f"min_pos_err={record.min_pos_error:.2f}")
            if record.min_angle_error is not None:
                parts.append(f"min_angle_err={record.min_angle_error:.3f}")
        if record.replans:
            parts.append(f"replans={record.replans}")
        if record.planning_time_s:
            parts.append(f"planning_s={record.planning_time_s:.2f}")
        print(" | ".join(parts))

    def _print_summary(self, summary: RunSummary, payload: dict[str, Any]) -> None:
        print(f"\n=== eval summary ({self.env_key}, {self.meta.get('protocol', '?')}) ===")
        print(
            f"success_rate: {summary.success_rate:.1f}% "
            f"({summary.num_successes}/{summary.num_episodes})"
        )
        print(f"mean_episode_length: {summary.mean_episode_length:.1f}")
        if summary.mean_steps_to_success is not None:
            print(f"mean_steps_to_success: {summary.mean_steps_to_success:.1f}")
        if summary.mean_min_state_distance is not None:
            std = summary.std_min_state_distance or 0.0
            print(
                f"mean_min_state_distance: {summary.mean_min_state_distance:.2f} "
                f"± {std:.2f}"
            )
        if summary.mean_final_state_distance is not None:
            print(f"mean_final_state_distance: {summary.mean_final_state_distance:.2f}")
        if self.config.log_planning:
            per_solve = (
                self.total_planning_time_s / self.total_replans
                if self.total_replans
                else 0.0
            )
            print(
                f"planning: {self.total_replans} solves, "
                f"{self.total_planning_time_s:.2f}s total "
                f"({per_solve:.3f}s/solve), "
                f"mean {summary.mean_planning_time_s:.3f}s/episode"
            )
        print(f"wall_time: {summary.wall_time_s:.1f}s")
        if "output_json" in payload:
            print(f"saved metrics: {payload['output_json']}")
        if "output_csv" in payload:
            print(f"saved episodes: {payload['output_csv']}")
