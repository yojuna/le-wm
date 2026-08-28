#!/usr/bin/env python3
"""Oracle test: PushT success fires when state matches goal_state.

No GPU / checkpoint required. Run from le-wm/:

    python scripts/test_pusht_success.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))
os.environ.setdefault("MUJOCO_GL", "egl")

from eval_logging.extractors import (  # noqa: E402
    PUSHT_ANGLE_TOL,
    PUSHT_POS_TOL,
    pusht_pose_errors,
    pusht_success,
)


def _env():
    import stable_worldmodel as swm

    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=1,
        max_episode_steps=50,
        image_shape=(224, 224),
    )
    return world


def _unwrapped(world):
    return world.envs.envs[0].unwrapped


def test_teleport_to_goal_succeeds(world) -> None:
    world.reset(seed=0)
    env = _unwrapped(world)
    goal = np.asarray(env.goal_state, dtype=np.float64).copy()

    env._set_goal_state(goal)
    env._set_state(goal)

    # One physics step with zero relative action to refresh obs / terminated
    action = np.zeros(world.envs.single_action_space.shape, dtype=np.float32)
    _obs, _reward, terminated, _truncated, info = world.envs.step(action[None, ...])

    state = np.asarray(info["state"][0]).reshape(-1)
    goal_state = np.asarray(info["goal_state"][0]).reshape(-1)
    pos_err, ang_err = pusht_pose_errors(goal_state, state)

    assert bool(np.asarray(terminated)[0]), (
        f"expected terminated=True after teleport; "
        f"pos_err={pos_err:.4f} (tol {PUSHT_POS_TOL}), "
        f"ang_err={ang_err:.4f} (tol {PUSHT_ANGLE_TOL})"
    )
    assert pusht_success(goal_state, state), (
        f"pusht_success False: pos_err={pos_err}, ang_err={ang_err}"
    )
    print(
        f"PASS teleport-to-goal: terminated=True "
        f"pos_err={pos_err:.4f} ang_err={ang_err:.4f}"
    )


def test_far_from_goal_fails(world) -> None:
    world.reset(seed=1)
    env = _unwrapped(world)
    goal = np.asarray(env.goal_state, dtype=np.float64).copy()
    far = goal.copy()
    far[:4] = far[:4] + 150.0  # well beyond pos tol 20

    env._set_goal_state(goal)
    env._set_state(far)

    action = np.zeros(world.envs.single_action_space.shape, dtype=np.float32)
    _obs, _reward, terminated, _truncated, info = world.envs.step(action[None, ...])

    state = np.asarray(info["state"][0]).reshape(-1)
    goal_state = np.asarray(info["goal_state"][0]).reshape(-1)
    pos_err, ang_err = pusht_pose_errors(goal_state, state)

    assert not bool(np.asarray(terminated)[0]), (
        f"expected terminated=False when far; pos_err={pos_err:.4f}"
    )
    assert not pusht_success(goal_state, state)
    print(
        f"PASS far-from-goal: terminated=False "
        f"pos_err={pos_err:.4f} ang_err={ang_err:.4f}"
    )


def test_wrapped_angle_equivalence() -> None:
    """Goal angle outside [0, 2π) must still match an equivalent wrapped state."""
    # state angle as stored by env obs: angle % 2π
    state = np.array([100.0, 100.0, 200.0, 200.0, 0.5, 0.0, 0.0])
    # same pose, goal angle = 0.5 - 2π (common when sampling [-2π, 2π])
    goal = state.copy()
    goal[4] = 0.5 - 2.0 * np.pi

    pos_err, ang_err = pusht_pose_errors(goal, state)
    assert pos_err < 1e-9, pos_err
    assert ang_err < 1e-9, ang_err
    assert pusht_success(goal, state)

    # Unfixed formula would yield negative angle_diff for large |Δ|
    goal_bad = state.copy()
    goal_bad[4] = state[4] + 3.0 * np.pi
    _pos, ang = pusht_pose_errors(goal_bad, state)
    assert ang >= 0.0, f"angle error must be non-negative, got {ang}"
    print(f"PASS wrapped-angle: ang_err={ang_err:.2e}, large-Δ ang={ang:.4f} >= 0")


def main() -> int:
    test_wrapped_angle_equivalence()
    world = _env()
    try:
        test_teleport_to_goal_succeeds(world)
        test_far_from_goal_fails(world)
    finally:
        world.close()
    print("All PushT success-criteria checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
