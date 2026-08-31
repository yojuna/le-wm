"""Live-bank oracle pairs: (t, t+window) windows from physics rollouts.

C0.3-redo ([docs/12a_c03_redo.md](../../docs/12a_c03_redo.md)): goal is the
state actually reached by the same rollout, so stored actions are a true
oracle for those (reachable) goals. Not kinematic FD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from eval_logging.extractors import pusht_pose_errors, pusht_success
from eval_logging.pairs import EvalPair, TrajectoryBank
from eval_setup import fit_process

SHORT_HORIZON_POS = (12.0, 25.0)
SHORT_HORIZON_ANG = 0.25
DEFAULT_WINDOW = 25
DEFAULT_STRIDE = 25


def block_xy_steps(path_state: np.ndarray) -> np.ndarray:
    """Per-step ‖Δblock_xy‖ along a window (L, ≥4)."""
    st = np.asarray(path_state, dtype=np.float64)
    if st.ndim != 2 or st.shape[0] < 2 or st.shape[-1] < 4:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(st[1:, 2:4] - st[:-1, 2:4], axis=-1)


def pair_block_step_median(pair: EvalPair) -> float:
    if pair.path_state is None:
        return 0.0
    d = block_xy_steps(pair.path_state)
    if d.size == 0:
        return 0.0
    return float(np.median(d))


def window_block_moving_pairs(
    bank: TrajectoryBank,
    *,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    scan_stride: int = 1,
    median_step_block_xy_min: float = 2.0,
    num_eval: int | None = None,
    seed: int = 0,
) -> list[EvalPair]:
    """25-step windows where the *typical* env step moves the block.

    No short_horizon pose band — that band selected near-settled hops
    (live-bank per-step block-xy median 0). This is a dynamics bank for B.eval-block.
    """
    raw: list[tuple[int, EvalPair]] = []
    for ep_i, ep in enumerate(bank.episodes):
        n = len(ep)
        if n < window + 1 or not ep.state or not ep.action or not ep.pixels:
            continue
        for start in range(0, n - window, scan_stride):
            goal = start + window
            st = np.stack(
                [
                    np.asarray(ep.state[i], dtype=np.float32).reshape(-1)
                    for i in range(start, goal + 1)
                ]
            )
            if float(np.median(block_xy_steps(st))) < float(median_step_block_xy_min):
                continue
            pos, _ang = pusht_pose_errors(ep.state[goal], ep.state[start])
            acts = np.stack(
                [
                    np.asarray(ep.action[i], dtype=np.float32).reshape(-1)
                    for i in range(start, goal)
                ],
                axis=0,
            )
            proprio_s = (
                np.asarray(ep.proprio[start]) if ep.proprio else np.asarray(ep.state[start])[:2]
            )
            proprio_g = (
                np.asarray(ep.proprio[goal]) if ep.proprio else np.asarray(ep.state[goal])[:2]
            )
            path_prop = None
            if ep.proprio:
                path_prop = np.stack(
                    [np.asarray(ep.proprio[i]).reshape(-1) for i in range(start, goal + 1)]
                )
            raw.append(
                (
                    ep_i,
                    EvalPair(
                        seed=ep.seed,
                        start_step=start,
                        init_pixels=np.asarray(ep.pixels[start], copy=True),
                        init_state=np.asarray(ep.state[start], copy=True),
                        init_proprio=np.asarray(proprio_s, copy=True),
                        goal_pixels=np.asarray(ep.pixels[goal], copy=True),
                        goal_state=np.asarray(ep.state[goal], copy=True),
                        goal_proprio=np.asarray(proprio_g, copy=True),
                        pos_progress=float(pos),
                        from_success_ep=ep.succeeded,
                        oracle_actions=acts,
                        path_pixels=np.stack(
                            [np.asarray(ep.pixels[i], copy=True) for i in range(start, goal + 1)]
                        ),
                        path_state=st,
                        path_proprio=path_prop,
                    ),
                )
            )
    candidates: list[EvalPair] = []
    last_ep, last_start = -1, -10**9
    for ep_i, pair in raw:
        if ep_i != last_ep:
            last_ep = ep_i
            last_start = -10**9
        if pair.start_step < last_start + stride:
            continue
        candidates.append(pair)
        last_start = pair.start_step
    if num_eval is None:
        return candidates
    if len(candidates) < num_eval:
        raise RuntimeError(
            f"only {len(candidates)} block-moving windows "
            f"(median Δblock_xy ≥ {median_step_block_xy_min}, raw {len(raw)}), "
            f"need {num_eval}"
        )
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(candidates), size=num_eval, replace=False))
    return [candidates[int(i)] for i in idx]


class OracleReplayPolicy:
    """Play stored env-scale actions for the current pair; no scaler."""

    def __init__(self, sequences: list[np.ndarray]):
        self.sequences = [np.asarray(s, dtype=np.float32) for s in sequences]
        self.pair_idx = 0
        self.t = 0
        self.env = None
        self.type = "oracle_replay"

    def set_env(self, env) -> None:
        self.env = env

    def begin_pair(self, pair_idx: int) -> None:
        self.pair_idx = int(pair_idx)
        self.t = 0

    def get_action(self, obs, **kwargs):
        seq = self.sequences[self.pair_idx]
        n_env = 1
        if self.env is not None:
            n_env = int(getattr(self.env, "num_envs", 1) or 1)
        dim = int(seq.shape[-1]) if seq.size else 2
        if self.t >= len(seq):
            a = np.zeros(dim, dtype=np.float32)
        else:
            a = np.asarray(seq[self.t], dtype=np.float32).reshape(-1)
            self.t += 1
        return np.broadcast_to(a, (n_env, dim)).copy()


def window_oracle_pairs(
    bank: TrajectoryBank,
    *,
    window: int = DEFAULT_WINDOW,
    stride: int = DEFAULT_STRIDE,
    scan_stride: int = 1,
    min_pos: float = SHORT_HORIZON_POS[0],
    max_pos: float = SHORT_HORIZON_POS[1],
    max_ang: float = SHORT_HORIZON_ANG,
    num_eval: int | None = None,
    seed: int = 0,
    drop_already_success: bool = True,
) -> list[EvalPair]:
    """(t, t+window) windows; actions[t:t+window] are from_state transitions.

    Scan every start (``scan_stride=1``) for paper-style short_horizon pose
    change, then keep non-overlapping windows (gap=``stride``). Raw stride-25
    hops on Weak/GoalPush are typically ~150 pose units and miss the band.
    Starts already inside env success tolerance are dropped so replay is not
    a free success at t=0.
    """
    raw: list[tuple[int, EvalPair]] = []
    n_already = 0
    for ep_i, ep in enumerate(bank.episodes):
        n = len(ep)
        if n < window + 1 or not ep.state or not ep.action or not ep.pixels:
            continue
        for start in range(0, n - window, scan_stride):
            goal = start + window
            pos, ang = pusht_pose_errors(ep.state[goal], ep.state[start])
            if pos < min_pos or pos > max_pos or ang > max_ang:
                continue
            if drop_already_success and pusht_success(ep.state[goal], ep.state[start]):
                n_already += 1
                continue
            acts = np.stack(
                [
                    np.asarray(ep.action[i], dtype=np.float32).reshape(-1)
                    for i in range(start, goal)
                ],
                axis=0,
            )
            proprio_s = (
                np.asarray(ep.proprio[start])
                if ep.proprio
                else np.asarray(ep.state[start])[:2]
            )
            proprio_g = (
                np.asarray(ep.proprio[goal])
                if ep.proprio
                else np.asarray(ep.state[goal])[:2]
            )
            path_prop = None
            if ep.proprio:
                path_prop = np.stack(
                    [np.asarray(ep.proprio[i]).reshape(-1) for i in range(start, goal + 1)]
                )
            raw.append(
                (
                    ep_i,
                    EvalPair(
                        seed=ep.seed,
                        start_step=start,
                        init_pixels=np.asarray(ep.pixels[start], copy=True),
                        init_state=np.asarray(ep.state[start], copy=True),
                        init_proprio=np.asarray(proprio_s, copy=True),
                        goal_pixels=np.asarray(ep.pixels[goal], copy=True),
                        goal_state=np.asarray(ep.state[goal], copy=True),
                        goal_proprio=np.asarray(proprio_g, copy=True),
                        pos_progress=float(pos),
                        from_success_ep=ep.succeeded,
                        oracle_actions=acts,
                        path_pixels=np.stack(
                            [np.asarray(ep.pixels[i], copy=True) for i in range(start, goal + 1)]
                        ),
                        path_state=np.stack(
                            [
                                np.asarray(ep.state[i], dtype=np.float32).reshape(-1)
                                for i in range(start, goal + 1)
                            ]
                        ),
                        path_proprio=path_prop,
                    ),
                )
            )
    # Greedy non-overlap within an episode (spec stride 25).
    candidates: list[EvalPair] = []
    last_ep, last_start = -1, -10**9
    for ep_i, pair in raw:
        if ep_i != last_ep:
            last_ep = ep_i
            last_start = -10**9
        if pair.start_step < last_start + stride:
            continue
        candidates.append(pair)
        last_start = pair.start_step
    if num_eval is None:
        return candidates
    if len(candidates) < num_eval:
        raise RuntimeError(
            f"only {len(candidates)} short_horizon oracle windows "
            f"(in-band raw {len(raw)}, dropped {n_already} already-success), "
            f"need {num_eval}"
        )
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(candidates), size=num_eval, replace=False))
    return [candidates[int(i)] for i in idx]


def save_oracle_bank(
    path: Path,
    pairs: list[EvalPair],
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    npz = path / "pairs.npz"
    n = len(pairs)
    window = (
        int(pairs[0].oracle_actions.shape[0])
        if n and pairs[0].oracle_actions is not None
        else 0
    )
    act_dim = int(pairs[0].oracle_actions.shape[-1]) if window else 2
    actions = np.zeros((n, window, act_dim), dtype=np.float32)
    for i, p in enumerate(pairs):
        if p.oracle_actions is not None:
            actions[i] = p.oracle_actions
    payload: dict[str, np.ndarray] = {
        "actions": actions,
        "init_pixels": np.stack([np.asarray(p.init_pixels) for p in pairs]),
        "goal_pixels": np.stack([np.asarray(p.goal_pixels) for p in pairs]),
        "init_state": np.stack([np.asarray(p.init_state).reshape(-1) for p in pairs]),
        "goal_state": np.stack([np.asarray(p.goal_state).reshape(-1) for p in pairs]),
        "init_proprio": np.stack([np.asarray(p.init_proprio).reshape(-1) for p in pairs]),
        "goal_proprio": np.stack([np.asarray(p.goal_proprio).reshape(-1) for p in pairs]),
        "pos_progress": np.asarray([p.pos_progress for p in pairs], dtype=np.float32),
        "ep_seed": np.asarray([p.seed for p in pairs], dtype=np.int32),
        "start_step": np.asarray([p.start_step for p in pairs], dtype=np.int32),
        "from_success_ep": np.asarray([p.from_success_ep for p in pairs], dtype=np.bool_),
    }
    if n and pairs[0].path_pixels is not None:
        payload["path_pixels"] = np.stack([np.asarray(p.path_pixels) for p in pairs])
    if n and pairs[0].path_state is not None:
        payload["path_state"] = np.stack([np.asarray(p.path_state) for p in pairs])
    if n and pairs[0].path_proprio is not None:
        payload["path_proprio"] = np.stack([np.asarray(p.path_proprio) for p in pairs])
    np.savez_compressed(npz, **payload)
    meta = {
        "n_pairs": n,
        "window": window,
        "pair_band": "short_horizon",
        "action_pack": "tile_block",
        **(extra or {}),
    }
    (path / "pairs.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return npz


def load_oracle_bank(path: Path) -> tuple[list[EvalPair], dict]:
    path = Path(path)
    npz = path / "pairs.npz" if path.is_dir() else path
    blob = np.load(npz, allow_pickle=True)
    meta_path = (
        npz.with_name("pairs.meta.json")
        if npz.name == "pairs.npz"
        else npz.with_suffix(".meta.json")
    )
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    n = int(blob["actions"].shape[0])
    has_path_pix = "path_pixels" in blob.files
    has_path_st = "path_state" in blob.files
    has_path_pr = "path_proprio" in blob.files
    pairs: list[EvalPair] = []
    for i in range(n):
        pairs.append(
            EvalPair(
                seed=int(blob["ep_seed"][i]),
                start_step=int(blob["start_step"][i]),
                init_pixels=np.asarray(blob["init_pixels"][i]),
                init_state=np.asarray(blob["init_state"][i]),
                init_proprio=np.asarray(blob["init_proprio"][i]),
                goal_pixels=np.asarray(blob["goal_pixels"][i]),
                goal_state=np.asarray(blob["goal_state"][i]),
                goal_proprio=np.asarray(blob["goal_proprio"][i]),
                pos_progress=float(blob["pos_progress"][i]),
                from_success_ep=bool(blob["from_success_ep"][i]),
                oracle_actions=np.asarray(blob["actions"][i]),
                path_pixels=np.asarray(blob["path_pixels"][i]) if has_path_pix else None,
                path_state=np.asarray(blob["path_state"][i]) if has_path_st else None,
                path_proprio=np.asarray(blob["path_proprio"][i]) if has_path_pr else None,
            )
        )
    return pairs, meta


def fit_process_from_oracle_pairs(pairs: list[EvalPair], keys_to_cache: list[str]) -> dict:
    """StandardScaler from the bank's own physics actions/states (not kinematic FD)."""
    columns: dict[str, np.ndarray] = {}
    for key in keys_to_cache:
        if key == "pixels":
            continue
        chunks: list[np.ndarray] = []
        for p in pairs:
            if key == "action" and p.oracle_actions is not None:
                chunks.append(np.asarray(p.oracle_actions, dtype=np.float64))
            elif key == "state":
                if p.path_state is not None:
                    chunks.append(np.asarray(p.path_state, dtype=np.float64))
                else:
                    chunks.append(
                        np.stack(
                            [
                                np.asarray(p.init_state).reshape(-1),
                                np.asarray(p.goal_state).reshape(-1),
                            ]
                        )
                    )
            elif key == "proprio":
                if p.path_proprio is not None:
                    chunks.append(np.asarray(p.path_proprio, dtype=np.float64))
                else:
                    chunks.append(
                        np.stack(
                            [
                                np.asarray(p.init_proprio).reshape(-1),
                                np.asarray(p.goal_proprio).reshape(-1),
                            ]
                        )
                    )
        if not chunks:
            raise KeyError(f"no {key!r} in oracle pairs for process fitting")
        columns[key] = np.concatenate(chunks, axis=0)
    return fit_process(keys_to_cache, columns)
