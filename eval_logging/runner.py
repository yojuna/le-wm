"""Instrumented World.evaluate loop with structured metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
import json

import numpy as np
from PIL import Image

from eval_logging.logger import EvalRunLogger
from eval_logging.pairs import EvalPair


class ActionRolloutRecorder:
    """Capture per-step physics actions + pixels during evaluate_goal_offset.

    Used by C0.3: privileged GoalPush/Weak on kinematic offset pairs.
    Must never be fed kinematic finite-difference actions.
    """

    def __init__(self):
        self.episodes: list[dict] = []
        self._actions: list[np.ndarray] = []
        self._pixels: list[np.ndarray] = []
        self._states: list[np.ndarray] = []
        self._policy = None

    def wrap_policy(self, policy):
        recorder = self
        inner_get = policy.get_action

        def get_action(obs, **kwargs):
            act = inner_get(obs, **kwargs)
            arr = np.asarray(act, copy=True)
            recorder._actions.append(np.asarray(arr[0] if arr.ndim > 1 else arr).reshape(-1))
            return act

        policy.get_action = get_action
        self._policy = policy
        return policy

    def on_step(self, world, env_idx: int = 0) -> None:
        infos = world.infos
        pixels = infos.get("pixels")
        if pixels is not None:
            frame = np.asarray(pixels[env_idx])
            if frame.ndim > 3:
                frame = frame[-1]
            self._pixels.append(np.asarray(frame, copy=True))
        state = infos.get("state")
        if state is not None:
            self._states.append(np.asarray(state[env_idx]).reshape(-1).copy())

    def end_episode(self) -> None:
        acts = (
            np.stack(self._actions, axis=0).astype(np.float32)
            if self._actions
            else np.zeros((0, 2), dtype=np.float32)
        )
        pix = (
            np.stack(self._pixels, axis=0)
            if self._pixels
            else np.zeros((0, 1, 1, 3), dtype=np.uint8)
        )
        st = (
            np.stack(self._states, axis=0).astype(np.float32)
            if self._states
            else np.zeros((0, 1), dtype=np.float32)
        )
        self.episodes.append({"action": acts, "pixels": pix, "state": st})
        self._actions = []
        self._pixels = []
        self._states = []

    def save(
        self,
        path: Path,
        *,
        pairs: Sequence[EvalPair],
        extra: dict | None = None,
        success: np.ndarray | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.episodes)
        lengths = np.asarray([len(ep["action"]) for ep in self.episodes], dtype=np.int32)
        max_t = int(lengths.max()) if n else 0
        act_dim = int(self.episodes[0]["action"].shape[-1]) if n and self.episodes[0]["action"].size else 2
        actions = np.zeros((n, max_t, act_dim), dtype=np.float32)
        for i, ep in enumerate(self.episodes):
            t = ep["action"].shape[0]
            if t:
                actions[i, :t] = ep["action"]
        payload = {
            "action": actions,
            "length": lengths,
            "success": np.asarray(success, dtype=np.bool_)
            if success is not None
            else np.zeros(n, dtype=np.bool_),
            "init_state": np.stack([np.asarray(p.init_state).reshape(-1) for p in pairs[:n]]),
            "goal_state": np.stack([np.asarray(p.goal_state).reshape(-1) for p in pairs[:n]]),
            "init_pixels": np.stack([np.asarray(p.init_pixels) for p in pairs[:n]]),
            "goal_pixels": np.stack([np.asarray(p.goal_pixels) for p in pairs[:n]]),
        }
        # per-episode pixel/state stacks are ragged; store as object arrays
        payload["rollout_pixels"] = np.array(
            [ep["pixels"] for ep in self.episodes], dtype=object
        )
        payload["rollout_state"] = np.array(
            [ep["state"] for ep in self.episodes], dtype=object
        )
        np.savez_compressed(path, **payload)
        meta = {"n_episodes": n, "max_t": max_t, **(extra or {})}
        path.with_suffix(".meta.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )


def load_oracle_rollouts(path: Path) -> dict:
    path = Path(path)
    blob = np.load(path, allow_pickle=True)
    data = {k: blob[k] for k in blob.files}
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        data["meta"] = json.loads(meta_path.read_text())
    else:
        data["meta"] = {}
    return data


def evaluate_logged(
    world,
    logger: EvalRunLogger,
    *,
    episodes: int,
    seed: int | None,
    on_step_extra: Callable[[Any], None] | None = None,
    render_dir: Path | None = None,
    render_fn: Callable[[], np.ndarray] | None = None,
    video_dir: Path | None = None,
    interrupt_exceptions: tuple[type[BaseException], ...] = (),
) -> dict:
    """Run world._run with per-episode metrics and optional frame dumps."""
    from stable_worldmodel.plot import save_video

    logger.begin_run()
    frame_idx = 0
    video_frames: dict[int, list] | None = defaultdict(list) if video_dir else None

    def on_step(w):
        nonlocal frame_idx
        for env_idx in range(w.num_envs):
            logger.on_step(w, env_idx)

        if video_frames is not None:
            for env_idx in range(w.num_envs):
                pixels = w.infos["pixels"][env_idx]
                frame = pixels[-1] if pixels.ndim > 3 else pixels
                video_frames[env_idx].append(np.asarray(frame).copy())

        if render_fn is not None and render_dir is not None:
            frame = render_fn()
            Image.fromarray(np.asarray(frame)).save(
                render_dir / f"frame_{frame_idx:05d}.png"
            )
            frame_idx += 1

        if on_step_extra is not None:
            on_step_extra(w)

    def on_done(env_idx, ep_idx, w):
        logger.on_episode_done(env_idx, ep_idx, w)
        if video_frames is not None and video_dir is not None:
            save_video(video_dir / f"episode_{ep_idx}.mp4", video_frames.pop(env_idx, []))

    interrupted = False
    try:
        world._run(
            episodes=episodes,
            seed=seed,
            on_step=on_step,
            on_done=on_done,
        )
    except interrupt_exceptions:
        print("evaluation interrupted — finalizing metrics")
        interrupted = True

    if render_dir is not None and frame_idx:
        print(f"saved {frame_idx} pixel frames to {render_dir}")

    if video_frames is not None and video_dir is not None:
        for env_idx, frames in video_frames.items():
            save_video(video_dir / f"episode_remaining_{env_idx}.mp4", frames)

    return logger.finalize(interrupted=interrupted)


def _as_info_tensor(value: np.ndarray, shape_prefix: tuple[int, ...]) -> np.ndarray:
    """Broadcast a single-env array into World infos shape (num_envs, T, ...)."""
    arr = np.asarray(value)
    return np.broadcast_to(arr[None, None, ...], shape_prefix + arr.shape).copy()


def _apply_pair_to_env(env, pair: EvalPair) -> None:
    env._set_state(pair.init_state)
    env._set_goal_state(pair.goal_state)


def _inject_pair_infos(world, pair: EvalPair) -> dict[str, np.ndarray]:
    """Write init + goal tensors into world.infos; return goal snapshot."""
    shape_prefix = world.infos["pixels"].shape[:2]

    world.infos["pixels"] = _as_info_tensor(pair.init_pixels, shape_prefix)
    if "state" in world.infos:
        world.infos["state"] = _as_info_tensor(pair.init_state, shape_prefix)
    if "proprio" in world.infos:
        world.infos["proprio"] = _as_info_tensor(pair.init_proprio, shape_prefix)

    goal_keys = {
        "goal": _as_info_tensor(pair.goal_pixels, shape_prefix),
        "goal_state": _as_info_tensor(pair.goal_state, shape_prefix),
        "goal_proprio": _as_info_tensor(pair.goal_proprio, shape_prefix),
    }
    for key, value in goal_keys.items():
        world.infos[key] = value
    return {k: v.copy() for k, v in goal_keys.items()}


def evaluate_goal_offset(
    world,
    logger: EvalRunLogger,
    pairs: Sequence[EvalPair],
    *,
    eval_budget: int,
    video_dir: Path | None = None,
    plan_debugger=None,
    interrupt_exceptions: tuple[type[BaseException], ...] = (),
    rollout_recorder: ActionRolloutRecorder | None = None,
) -> dict:
    """Paper-style eval: fixed start/goal pairs, budget-capped wait rollouts.

    Mirrors ``World._evaluate_from_dataset`` without an HDF5 dataset:
    set physical state/goal, inject goal pixels, reinject goals each step,
    run ``max_steps=eval_budget`` in wait mode.
    """
    from stable_worldmodel.plot import save_video

    if world.num_envs != 1:
        raise ValueError("evaluate_goal_offset currently requires num_envs=1")

    logger.begin_run()
    interrupted = False
    video_frames: list | None = [] if video_dir else None

    try:
        for ep_idx, pair in enumerate(pairs):
            # C1: drop cached goal emb between pairs; set explicit cache key on
            # the model (not world.infos — CEM only expands tensors safely).
            policy = getattr(world, "policy", None)
            model = getattr(getattr(policy, "solver", None), "model", None)
            if model is not None and hasattr(model, "clear_goal_cache"):
                model.clear_goal_cache()
                model._forced_goal_cache_key = f"pair:{ep_idx}:seed:{pair.seed}"
            if hasattr(policy, "begin_pair"):
                policy.begin_pair(ep_idx)

            world.reset(seed=pair.seed)
            env = world.envs.envs[0].unwrapped
            _apply_pair_to_env(env, pair)

            world.terminateds = np.zeros(world.num_envs, dtype=bool)
            world.truncateds = np.zeros(world.num_envs, dtype=bool)
            goal_snapshot = _inject_pair_infos(world, pair)

            if plan_debugger is not None:
                plan_debugger.begin_episode(ep_idx, pair.seed, world.infos)

            saw_success = False
            episode_done = False

            def on_step(w, _snap=goal_snapshot):
                nonlocal saw_success
                w.infos.update(deepcopy(_snap))
                saw_success = saw_success or bool(w.terminateds[0])
                logger.on_step(w, 0)
                if rollout_recorder is not None:
                    rollout_recorder.on_step(w, 0)
                if plan_debugger is not None:
                    plan_debugger.on_step(w, 0)
                if video_frames is not None:
                    pixels = w.infos["pixels"][0]
                    frame = pixels[-1] if pixels.ndim > 3 else pixels
                    video_frames.append(np.asarray(frame).copy())

            def on_done(env_idx, _ep_count, w):
                nonlocal episode_done
                episode_done = True
                if saw_success:
                    w.terminateds[env_idx] = True
                logger.on_episode_done(env_idx, ep_idx, w)
                if rollout_recorder is not None:
                    rollout_recorder.end_episode()
                if plan_debugger is not None:
                    plan_debugger.end_episode(
                        success=bool(w.terminateds[env_idx]),
                        truncated=bool(w.truncateds[env_idx]),
                    )
                if video_dir is not None and video_frames is not None:
                    save_video(video_dir / f"episode_{ep_idx}.mp4", list(video_frames))
                    video_frames.clear()

            world._run(
                episodes=1,
                max_steps=eval_budget,
                mode="wait",
                on_step=on_step,
                on_done=on_done,
            )

            # Budget exhausted without terminate/truncate → finalize manually
            if not episode_done:
                if saw_success:
                    world.terminateds[0] = True
                    world.truncateds[0] = False
                else:
                    world.truncateds[0] = True
                logger.on_episode_done(0, ep_idx, world)
                if rollout_recorder is not None:
                    rollout_recorder.end_episode()
                if plan_debugger is not None:
                    plan_debugger.end_episode(
                        success=bool(world.terminateds[0]),
                        truncated=bool(world.truncateds[0]),
                    )
                if video_dir is not None and video_frames is not None:
                    save_video(video_dir / f"episode_{ep_idx}.mp4", list(video_frames))
                    video_frames.clear()

            # Clear MPC buffer between pairs (same intent as _needs_flush)
            if hasattr(world.policy, "_action_buffer"):
                for buf in world.policy._action_buffer:
                    buf.clear()
                if getattr(world.policy, "_next_init", None) is not None:
                    world.policy._next_init = None

    except interrupt_exceptions:
        print("evaluation interrupted — finalizing metrics")
        interrupted = True

    if plan_debugger is not None:
        plan_debugger.finalize()

    return logger.finalize(interrupted=interrupted)
