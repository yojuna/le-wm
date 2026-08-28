"""Online trajectory bank standing in for HDF5 expert chunks.

Collects PushT rollouts (WeakPolicy or GoalPushPolicy), then samples
(start, start+goal_offset) pairs the same way eval.py samples from dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eval_logging.extractors import pusht_pose_errors, pusht_success
from eval_setup import _sample_actions, fit_process


@dataclass
class EpisodeTraj:
    seed: int
    pixels: list[np.ndarray] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    proprio: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    succeeded: bool = False

    def __len__(self) -> int:
        return len(self.state)


@dataclass
class EvalPair:
    """One paper-style start/goal pair (goal = start + offset)."""

    seed: int
    start_step: int
    init_pixels: np.ndarray
    init_state: np.ndarray
    init_proprio: np.ndarray
    goal_pixels: np.ndarray
    goal_state: np.ndarray
    goal_proprio: np.ndarray
    pos_progress: float = 0.0
    from_success_ep: bool = False


@dataclass
class TrajectoryBank:
    episodes: list[EpisodeTraj] = field(default_factory=list)
    env_name: str = ""
    collector: str = ""

    @property
    def num_steps(self) -> int:
        return sum(len(ep) for ep in self.episodes)

    @property
    def num_success_episodes(self) -> int:
        return sum(1 for ep in self.episodes if ep.succeeded)

    def columns(self, keys: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, list[np.ndarray]] = {k: [] for k in keys if k != "pixels"}
        for ep in self.episodes:
            for key in out:
                seq = getattr(ep, key, None)
                if not seq:
                    continue
                arr = np.stack(seq, axis=0)
                if arr.ndim == 1:
                    arr = arr[:, None]
                out[key].append(arr)
        return {k: np.concatenate(v, axis=0) for k, v in out.items() if v}


def _squeeze_frame(value: np.ndarray, env_idx: int = 0) -> np.ndarray:
    arr = np.asarray(value[env_idx])
    if arr.ndim >= 3 and arr.shape[0] <= 8:
        return np.asarray(arr[-1], copy=True)
    return np.asarray(arr, copy=True)


def _vector(value: np.ndarray, env_idx: int = 0) -> np.ndarray:
    arr = np.asarray(value[env_idx]).reshape(-1)
    return arr.astype(np.float64, copy=True)


def _make_collection_policy(world, *, env_name: str, seed: int, collector: str):
    if "PushT" not in env_name:
        return None
    if collector in ("weak", "weak_policy"):
        from stable_worldmodel.envs.pusht.expert_policy import WeakPolicy

        policy = WeakPolicy(seed=seed)
        policy.set_env(world.envs)
        return policy
    if collector in ("goal", "goal_push", "strong"):
        from eval_logging.collect_policy import GoalPushPolicy

        policy = GoalPushPolicy(seed=seed)
        policy.set_env(world.envs)
        return policy
    if collector in ("kinematic", "kin"):
        return None  # handled by collect_kinematic_bank
    raise ValueError(
        f"unknown collector={collector!r}; use 'kinematic', 'goal', or 'weak'"
    )


def collect_kinematic_bank(
    world,
    *,
    num_episodes: int,
    seed: int,
    env_name: str,
    horizon: int = 80,
    noise_std: float = 0.5,
) -> TrajectoryBank:
    """Build eval pairs via start→goal_state kinematic interpolation + render."""
    from eval_logging.collect_policy import kinematic_episode

    if "PushT" not in env_name:
        raise ValueError("kinematic collector is PushT-only")

    bank = TrajectoryBank(env_name=env_name, collector="kinematic")
    rng = np.random.default_rng(seed)
    env = world.envs.envs[0].unwrapped

    for ep_i in range(num_episodes):
        ep_seed = seed + ep_i
        world.reset(seed=ep_seed)
        data = kinematic_episode(
            env, horizon=horizon, noise_std=noise_std, rng=rng
        )
        traj = EpisodeTraj(
            seed=ep_seed,
            pixels=data["pixels"],
            state=data["state"],
            proprio=data["proprio"],
            action=data["action"],
            succeeded=data["succeeded"],
        )
        bank.episodes.append(traj)

    # Refresh world infos after kinematic writes
    world.reset(seed=seed + num_episodes)
    return bank


def collect_trajectory_bank(
    world,
    *,
    num_steps: int,
    seed: int,
    env_name: str,
    min_episode_len: int = 1,
    collector: str = "kinematic",
    num_episodes: int | None = None,
    kinematic_horizon: int = 80,
) -> TrajectoryBank:
    """Roll out a collection policy and store per-step observations."""
    if collector in ("kinematic", "kin"):
        # Enough episodes for pair sampling; ignore num_steps as primary budget
        n_eps = num_episodes or max(32, (num_steps // kinematic_horizon) + 4)
        return collect_kinematic_bank(
            world,
            num_episodes=n_eps,
            seed=seed,
            env_name=env_name,
            horizon=kinematic_horizon,
        )

    bank = TrajectoryBank(env_name=env_name, collector=collector)
    rng = np.random.default_rng(seed)
    collection_policy = _make_collection_policy(
        world, env_name=env_name, seed=seed, collector=collector
    )

    world.reset(seed=seed)
    current = EpisodeTraj(seed=int(world.envs.seeds[0]))
    steps_done = 0

    while steps_done < num_steps:
        infos = world.infos
        if "state" in infos:
            current.state.append(_vector(infos["state"]))
        if "proprio" in infos:
            current.proprio.append(_vector(infos["proprio"]))
        if "pixels" in infos:
            current.pixels.append(_squeeze_frame(infos["pixels"]))

        if collection_policy is not None:
            actions = collection_policy.get_action(infos)
        else:
            actions = _sample_actions(world, rng)

        current.action.append(np.asarray(actions[0], dtype=np.float64).reshape(-1))

        _, _, terminated, truncated, infos = world.envs.step(actions)
        world.infos = infos
        world.terminateds = terminated
        world.truncateds = truncated
        steps_done += 1

        if bool(np.asarray(terminated)[0]):
            current.succeeded = True

        done = bool(np.asarray(terminated)[0] or np.asarray(truncated)[0])
        if done or steps_done >= num_steps:
            if len(current) >= min_episode_len:
                bank.episodes.append(current)
            if steps_done >= num_steps:
                break
            next_seed = seed + len(bank.episodes)
            _, infos = world.envs.reset(seed=[next_seed])
            world.infos = infos
            world.terminateds = np.zeros(world.num_envs, dtype=bool)
            world.truncateds = np.zeros(world.num_envs, dtype=bool)
            current = EpisodeTraj(seed=int(world.envs.seeds[0]))

    return bank


def fit_process_from_bank(bank: TrajectoryBank, keys_to_cache: list[str]) -> dict:
    columns = bank.columns(keys_to_cache)
    return fit_process(keys_to_cache, columns)


def _pair_progress(ep: EpisodeTraj, start: int, goal_offset: int) -> float:
    goal_i = start + goal_offset
    pos0, _ = pusht_pose_errors(ep.state[goal_i], ep.state[start])
    # progress = how much closer start→goal moved relative to staying put
    # Using goal as reference: start distance to final pose vs 0
    return float(pos0)


def sample_eval_pairs(
    bank: TrajectoryBank,
    *,
    num_eval: int,
    goal_offset: int,
    seed: int,
    prefer_success: bool = True,
    min_pos_delta: float = 15.0,
    max_pos_delta: float = 55.0,
    max_angle_delta: float = 0.6,
    prefer_late: bool = False,
    mode: str = "offset",
) -> list[EvalPair]:
    """Sample eval pairs.

    Modes
    -----
    offset:
        Classic (start, start+goal_offset) windows — same structure as eval.py.
    short_horizon:
        Offset windows with small pose/angle change (paper-like reachable hops).
    finish:
        Start already near the episode end goal; goal is the final frame.
        Tests whether the planner can close out a near-complete configuration.
    """
    if mode == "short_horizon":
        min_pos_delta, max_pos_delta = 12.0, 25.0
        max_angle_delta = 0.25
        prefer_late = True
    elif mode == "finish":
        return _sample_finish_pairs(
            bank,
            num_eval=num_eval,
            seed=seed,
            min_start_pos_err=25.0,
            max_start_pos_err=45.0,
        )
    elif mode != "offset":
        raise ValueError(f"unknown pair mode={mode!r}")

    candidates: list[tuple[float, int, int, float]] = []

    for ep_i, ep in enumerate(bank.episodes):
        max_start = len(ep) - goal_offset - 1
        if max_start < 0:
            continue
        for start in range(max_start + 1):
            goal_i = start + goal_offset
            pos_delta, ang_delta = pusht_pose_errors(ep.state[goal_i], ep.state[start])
            if pos_delta < min_pos_delta or pos_delta > max_pos_delta:
                continue
            if ang_delta > max_angle_delta:
                continue
            mid = 0.5 * (min_pos_delta + max_pos_delta)
            band_score = 50.0 - abs(pos_delta - mid) - 10.0 * ang_delta
            if prefer_late:
                band_score += 30.0 * (start / max(max_start, 1))
            score = (1000.0 if ep.succeeded else 0.0) + band_score
            if prefer_success and not ep.succeeded:
                score *= 0.25
            candidates.append((score, ep_i, start, float(pos_delta)))

    if not candidates:
        for ep_i, ep in enumerate(bank.episodes):
            max_start = len(ep) - goal_offset - 1
            if max_start < 0:
                continue
            for start in range(max_start + 1):
                pos_delta, _ = pusht_pose_errors(
                    ep.state[start + goal_offset], ep.state[start]
                )
                if pos_delta < min_pos_delta:
                    continue
                candidates.append((pos_delta, ep_i, start, float(pos_delta)))

    if not candidates:
        raise RuntimeError(
            f"no valid start/goal pairs: need episodes longer than "
            f"goal_offset+1={goal_offset + 1}; bank has "
            f"{len(bank.episodes)} eps, {bank.num_steps} steps"
        )
    if len(candidates) < num_eval:
        raise RuntimeError(
            f"only {len(candidates)} valid pairs, need num_eval={num_eval}; "
            f"collect more steps or widen pos-delta band (mode={mode})"
        )

    rng = np.random.default_rng(seed)
    scores = np.asarray([c[0] for c in candidates], dtype=np.float64)
    scores = np.maximum(scores, 1e-6)
    probs = scores / scores.sum()
    chosen = rng.choice(len(candidates), size=num_eval, replace=False, p=probs)
    chosen = np.sort(chosen)

    pairs: list[EvalPair] = []
    for idx in chosen:
        _score, ep_i, start, pos_delta = candidates[int(idx)]
        ep = bank.episodes[ep_i]
        goal_i = start + goal_offset
        pairs.append(
            EvalPair(
                seed=ep.seed,
                start_step=start,
                init_pixels=np.asarray(ep.pixels[start], copy=True),
                init_state=np.asarray(ep.state[start], copy=True),
                init_proprio=np.asarray(ep.proprio[start], copy=True),
                goal_pixels=np.asarray(ep.pixels[goal_i], copy=True),
                goal_state=np.asarray(ep.state[goal_i], copy=True),
                goal_proprio=np.asarray(ep.proprio[goal_i], copy=True),
                pos_progress=pos_delta,
                from_success_ep=ep.succeeded,
            )
        )
    return pairs


def _sample_finish_pairs(
    bank: TrajectoryBank,
    *,
    num_eval: int,
    seed: int,
    min_start_pos_err: float,
    max_start_pos_err: float,
) -> list[EvalPair]:
    """Start near the final goal; goal = last frame of the episode."""
    candidates: list[tuple[float, int, int, float]] = []
    for ep_i, ep in enumerate(bank.episodes):
        if len(ep) < 5:
            continue
        goal_i = len(ep) - 1
        goal_state = ep.state[goal_i]
        for start in range(0, goal_i):
            pos_err, ang_err = pusht_pose_errors(goal_state, ep.state[start])
            if pos_err < min_start_pos_err or pos_err > max_start_pos_err:
                continue
            if ang_err > 0.5:
                continue
            # Never credit an already-solved start as a "finish" success.
            if pusht_success(goal_state, ep.state[start]):
                continue
            score = (1000.0 if ep.succeeded else 0.0) + (50.0 - pos_err) - 5.0 * ang_err
            candidates.append((score, ep_i, start, float(pos_err)))

    if len(candidates) < num_eval:
        raise RuntimeError(
            f"only {len(candidates)} finish-line pairs "
            f"(need start pos_err in [{min_start_pos_err}, {max_start_pos_err}]); "
            f"got num_eval={num_eval}"
        )

    rng = np.random.default_rng(seed)
    scores = np.asarray([c[0] for c in candidates], dtype=np.float64)
    scores = np.maximum(scores, 1e-6)
    probs = scores / scores.sum()
    chosen = rng.choice(len(candidates), size=num_eval, replace=False, p=probs)
    chosen = np.sort(chosen)

    pairs: list[EvalPair] = []
    for idx in chosen:
        _score, ep_i, start, pos_err = candidates[int(idx)]
        ep = bank.episodes[ep_i]
        goal_i = len(ep) - 1
        pairs.append(
            EvalPair(
                seed=ep.seed,
                start_step=start,
                init_pixels=np.asarray(ep.pixels[start], copy=True),
                init_state=np.asarray(ep.state[start], copy=True),
                init_proprio=np.asarray(ep.proprio[start], copy=True),
                goal_pixels=np.asarray(ep.pixels[goal_i], copy=True),
                goal_state=np.asarray(ep.state[goal_i], copy=True),
                goal_proprio=np.asarray(ep.proprio[goal_i], copy=True),
                pos_progress=pos_err,
                from_success_ep=ep.succeeded,
            )
        )
    return pairs


def bank_success_report(bank: TrajectoryBank) -> dict:
    n = len(bank.episodes)
    n_ok = bank.num_success_episodes
    lengths = [len(ep) for ep in bank.episodes]
    return {
        "collector": bank.collector,
        "episodes": n,
        "success_episodes": n_ok,
        "success_rate_pct": (100.0 * n_ok / n) if n else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "steps": bank.num_steps,
    }
