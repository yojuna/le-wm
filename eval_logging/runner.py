"""Instrumented World.evaluate loop with structured metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from eval_logging.logger import EvalRunLogger
from eval_logging.pairs import EvalPair


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
