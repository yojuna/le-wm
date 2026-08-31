"""Per-environment step metrics from World info dicts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

PUSHT_POS_TOL = 20.0
PUSHT_ANGLE_TOL = np.pi / 9


@dataclass(frozen=True)
class StepMetrics:
    """Scalar metrics extracted from one env step."""

    reward: float | None = None
    state_distance: float | None = None
    pos_error: float | None = None
    angle_error: float | None = None
    task_score: float | None = None
    n_contacts: int | None = None


def _env_scalar(infos: dict[str, Any], key: str, env_idx: int) -> float | None:
    if key not in infos:
        return None
    value = infos[key]
    if isinstance(value, np.ndarray):
        arr = np.asarray(value[env_idx]).reshape(-1)
        if arr.size == 0:
            return None
        scalar = arr[0]
        if np.isnan(scalar):
            return None
        return float(scalar)
    if isinstance(value, (list, tuple)):
        item = value[env_idx]
        if item is None or (isinstance(item, float) and np.isnan(item)):
            return None
        return float(item)
    return float(value)


def _env_vector(infos: dict[str, Any], key: str, env_idx: int) -> np.ndarray | None:
    if key not in infos:
        return None
    value = infos[key]
    if isinstance(value, np.ndarray):
        arr = np.asarray(value[env_idx]).reshape(-1)
        return arr.astype(np.float64, copy=False)
    return None


def _wrap_angle(angle: float) -> float:
    """Map angle into [0, 2π)."""
    two_pi = 2.0 * np.pi
    return float(np.mod(angle, two_pi))


def pusht_pose_errors(goal_state: np.ndarray, state: np.ndarray) -> tuple[float, float]:
    pos_diff = float(np.linalg.norm(goal_state[:4] - state[:4]))
    # Wrap before comparing — raw goal angles may be in [-2π, 2π] while
    # observations store angle % 2π; unwrapped |Δ| can exceed 2π and break
    # the classic min(|Δ|, 2π-|Δ|) formula (negative "errors").
    angle_diff = float(np.abs(_wrap_angle(goal_state[4]) - _wrap_angle(state[4])))
    angle_diff = float(np.minimum(angle_diff, 2.0 * np.pi - angle_diff))
    return pos_diff, angle_diff


def pusht_success(goal_state: np.ndarray, state: np.ndarray) -> bool:
    pos_diff, angle_diff = pusht_pose_errors(goal_state, state)
    return pos_diff < PUSHT_POS_TOL and angle_diff < PUSHT_ANGLE_TOL


def extract_step_metrics(env_key: str, world, env_idx: int) -> StepMetrics:
    infos = world.infos
    reward = _env_scalar(infos, "reward", env_idx)

    if env_key == "pusht":
        state = _env_vector(infos, "state", env_idx)
        goal_state = _env_vector(infos, "goal_state", env_idx)
        if state is not None and goal_state is not None:
            pos_error, angle_error = pusht_pose_errors(goal_state, state)
            state_distance = float(np.linalg.norm(goal_state - state))
            return StepMetrics(
                reward=reward if reward is not None else -state_distance,
                state_distance=state_distance,
                pos_error=pos_error,
                angle_error=angle_error,
                n_contacts=int(_env_scalar(infos, "n_contacts", env_idx) or 0),
            )

    if env_key == "reacher":
        score = _env_scalar(infos, "score", env_idx)
        finger = _env_vector(infos, "finger_pos", env_idx)
        target = _env_vector(infos, "target_pos", env_idx)
        state_distance = None
        if finger is not None and target is not None and finger.size >= 2:
            state_distance = float(np.linalg.norm(finger[:2] - target[:2]))
        return StepMetrics(
            reward=reward,
            state_distance=state_distance,
            task_score=score,
        )

    if reward is not None and reward <= 0:
        return StepMetrics(reward=reward, state_distance=-reward)
    return StepMetrics(reward=reward)
