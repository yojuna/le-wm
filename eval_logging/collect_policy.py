"""PushT collection policies for online_offset eval pairs.

``GoalPushPolicy`` — real-physics heuristic (may rarely succeed).
``KinematicGoalCollector`` — interpolates start→goal_state via ``_set_state``
and records rendered frames + finite-difference actions. This produces
reachable (t, t+offset) pairs similar in *structure* to expert HDF5 segments
used by eval.py, without downloading the dataset.
"""

from __future__ import annotations

import numpy as np
from stable_worldmodel.policy import BasePolicy


def _wrap_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _interp_angle(a0: float, a1: float, alpha: float) -> float:
    return float(a0 + alpha * _wrap_pi(a1 - a0))


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


class GoalPushPolicy(BasePolicy):
    """Heuristic real-physics pusher (best-effort; prefer kinematic for pairs)."""

    def __init__(
        self,
        seed: int | None = None,
        *,
        approach_dist: float = 22.0,
        contact_radius: float = 45.0,
        noise_std: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.approach_dist = approach_dist
        self.contact_radius = contact_radius
        self.noise_std = noise_std
        self.set_seed(seed)

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env) -> None:
        self.env = env

    def _envs(self):
        if hasattr(self.env, "envs"):
            return [e.unwrapped for e in self.env.envs]
        base = self.env.unwrapped
        if hasattr(base, "envs"):
            return [e.unwrapped for e in base.envs]
        return [base]

    def get_action(self, info_dict, **kwargs):
        envs = self._envs()
        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)
        states = np.asarray(info_dict["state"])
        goals = np.asarray(info_dict["goal_state"])

        for i, env in enumerate(envs):
            state = np.asarray(states[i])
            goal = np.asarray(goals[i])
            if state.ndim > 1:
                state = state[-1]
            if goal.ndim > 1:
                goal = goal[-1]
            state = state.reshape(-1)
            goal = goal.reshape(-1)

            agent, block = state[:2], state[2:4]
            g_agent, g_block = goal[:2], goal[2:4]
            d = g_block - block
            dist = float(np.linalg.norm(d)) + 1e-8
            direction = d / dist
            contact = block - direction * self.contact_radius
            if dist < 12.0:
                target = g_agent
            elif float(np.linalg.norm(agent - contact)) > self.approach_dist:
                target = contact
            else:
                target = block + direction * self.contact_radius

            delta = (target - agent) / float(env.action_scale)
            if self.noise_std > 0:
                delta = delta + self.rng.normal(0.0, self.noise_std, size=2)
            actions[i] = np.clip(delta, -1.0, 1.0).astype(np.float32)
        return actions


def kinematic_episode(
    env,
    *,
    horizon: int = 80,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Build one start→goal kinematic trajectory on an unwrapped PushT env.

    Returns dict with lists: pixels, state, proprio, action, and succeeded bool.
    """
    rng = rng or np.random.default_rng(0)
    start = np.asarray(env._get_obs(), dtype=np.float64).copy()
    goal = np.asarray(env.goal_state, dtype=np.float64).copy()

    pixels, states, proprios, actions = [], [], [], []
    prev_agent = start[:2].copy()

    for i in range(horizon):
        alpha = _smoothstep(i / max(horizon - 1, 1))
        state = (1.0 - alpha) * start + alpha * goal
        state[4] = _interp_angle(float(start[4]), float(goal[4]), alpha)
        if noise_std > 0 and i not in (0, horizon - 1):
            state[:4] = state[:4] + rng.normal(0.0, noise_std, size=4)

        env._set_state(state)
        obs = np.asarray(env._get_obs(), dtype=np.float64)
        proprio = np.concatenate((obs[:2], obs[-2:]))
        frame = env.render()

        agent = obs[:2]
        delta = (agent - prev_agent) / float(env.action_scale)
        # first frame: zero action placeholder (not used as transition)
        action = np.clip(delta, -1.0, 1.0).astype(np.float64)
        if i == 0:
            action = np.zeros(2, dtype=np.float64)

        pixels.append(np.asarray(frame, copy=True))
        states.append(obs.copy())
        proprios.append(proprio.copy())
        actions.append(action)
        prev_agent = agent.copy()

    # Ensure final frame is exactly the goal for success labeling
    env._set_state(goal)
    final = np.asarray(env._get_obs(), dtype=np.float64)
    from eval_logging.extractors import pusht_success

    succeeded = pusht_success(goal, final)
    return {
        "pixels": pixels,
        "state": states,
        "proprio": proprios,
        "action": actions,
        "succeeded": bool(succeeded),
    }
