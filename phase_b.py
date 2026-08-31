"""Phase B shared dump / encode / imagine helpers.

See docs/09_phase_b_plan.md. Latents were not persisted by offset_autopsy.py;
this module is the on-disk format for B1 probes and B2 drift curves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision.transforms import v2 as transforms
import stable_pretraining as spt

from eval_logging.extractors import pusht_pose_errors
from eval_logging.pairs import EpisodeTraj
from phi_data import frame_to_tensor

HISTORY = 3
ACTION_DIM = 10
ACTION_BLOCK = 5  # CEM / training frameskip; token dim = env_action * block
CEM_HORIZON = 5  # EnvSpec / pusht.yaml default; mark on drift plots
# CEM Box bounds are tiled with tensor.repeat(action_block) → [a0,a1]×5.
# Dump tokens use the same tile so AdaLN does not see zero-pad OOD.
ACTION_PACK = "tile_block"

PUSHT_FACTORS = (
    "agent_x",
    "agent_y",
    "block_x",
    "block_y",
    "block_angle",
    "agent_vx",
    "agent_vy",
)

REACHER_FACTORS = (
    "qpos_0",
    "qpos_1",
    "qvel_0",
    "qvel_1",
    "finger_x",
    "finger_y",
    "target_x",
    "target_y",
)

DUMP_VERSION = 2


def img_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def pad_action(a: np.ndarray, action_dim: int = ACTION_DIM) -> np.ndarray:
    """Zero-pad env action to action_encoder dim. Prefer pack_action_token."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    out = np.zeros(action_dim, dtype=np.float32)
    out[: min(len(a), action_dim)] = a[:action_dim]
    return out


def pack_action_token(
    a: np.ndarray,
    *,
    action_block: int = ACTION_BLOCK,
    action_dim: int = ACTION_DIM,
) -> np.ndarray:
    """Pack a raw env action into one LeWM action-encoder token.

    CEM flattened dim is ``env_dim * action_block``. Installed SWM tiles Box
    bounds with ``tensor.repeat(action_block)``, i.e. ``[ax, ay]`` repeated
    five times → length 10. Zero-padding dims 2–9 is OOD relative to that.
    """
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return np.zeros(action_dim, dtype=np.float32)
    if a.size >= action_dim:
        return a[:action_dim].copy()
    tiled = np.tile(a, int(action_block))
    if tiled.size < action_dim:
        out = np.zeros(action_dim, dtype=np.float32)
        out[: tiled.size] = tiled
        return out
    return tiled[:action_dim].astype(np.float32, copy=False)


def actions_to_tokens(actions: np.ndarray) -> np.ndarray:
    """(L, A_env|A_token) → (L, ACTION_DIM). Skip re-tile if already packed."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    if actions.shape[-1] == ACTION_DIM:
        return actions
    return np.stack([pack_action_token(a) for a in actions], axis=0)


def collector_uses_set_state(collector: str) -> bool:
    return collector in ("kinematic", "kin")


def action_convention_for_collector(collector: str) -> str:
    if collector_uses_set_state(collector):
        return "into_state_fd"
    return "from_state_step"


def resolve_dump_collector(
    env: str,
    *,
    collector: str = "",
    action_mode: str = "",
) -> str:
    """Resolve latent_dump collector. ``--action-mode diverse`` → random."""
    mode = (action_mode or "").strip().lower()
    if mode:
        aliases = {
            "diverse": "random",
            "random": "random",
            "kinematic": "kinematic",
            "kin": "kinematic",
            "weak": "weak",
            "goal": "goal",
            "goal_push": "goal",
        }
        if mode not in aliases:
            raise ValueError(
                f"unknown --action-mode {action_mode!r}; "
                "use diverse, random, kinematic, weak, or goal"
            )
        return aliases[mode]
    col = (collector or "").strip().lower()
    if col:
        if col in ("diverse",):
            return "random"
        return col
    return "kinematic" if env == "pusht" else "random"


def dump_default_out_dir(root: Path, env: str, collector: str, seed: int) -> Path:
    folder = "phase_b_dump_diverse" if collector == "random" else "phase_b_dump"
    return Path(root) / "eval_results" / env / folder / f"seed{seed}"


def validate_oracle_actor(actor: str) -> str:
    """C0.3 actors must take real env.step actions, not kinematic FD."""
    name = (actor or "").strip().lower()
    if name in ("cem_l2",):
        return "cem"
    if name in ("kinematic", "kin"):
        raise ValueError(
            f"oracle actor {actor!r} is invalid: do not replay kinematic "
            "finite-difference / _set_state actions"
        )
    if name not in ("goal_push", "weak", "cem", "oracle_replay"):
        raise ValueError(
            f"unknown actor {actor!r}; use goal_push, weak, cem, cem_l2, or oracle_replay"
        )
    return name


def effective_rank(x: np.ndarray, *, eps: float = 1e-12) -> float:
    """Participation ratio of covariance eigenvalues (LeVLJEPA-style)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 1:
        return float("nan")
    centered = arr - arr.mean(axis=0, keepdims=True)
    # (D, D) covariance; eigvalsh is ascending
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        eig = np.array([float(cov)], dtype=np.float64)
    else:
        eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 0.0)
    total = float(eig.sum())
    if total < eps:
        return 0.0
    p = eig / total
    return float(1.0 / np.sum(p * p))


def factor_names_for_env(env: str, state_dim: int) -> tuple[str, ...]:
    if env == "pusht":
        names = list(PUSHT_FACTORS[:state_dim])
        while len(names) < state_dim:
            names.append(f"state_{len(names)}")
        return tuple(names)
    # Reacher dump concat: qpos + qvel + finger_pos + target_pos (pairs.py)
    names = list(REACHER_FACTORS[:state_dim])
    while len(names) < state_dim:
        names.append(f"factor_{len(names)}")
    return tuple(names)


def remaining_pose_error(env: str, states: np.ndarray) -> np.ndarray:
    """Distance from each frame to the last frame of the segment."""
    goal = states[-1]
    out = np.zeros(len(states), dtype=np.float64)
    for i, st in enumerate(states):
        if env == "pusht" and st.size >= 5:
            pos, _ang = pusht_pose_errors(goal, st)
            out[i] = pos
        else:
            n = min(st.size, goal.size, 2)
            out[i] = float(np.linalg.norm(st[:n] - goal[:n]))
    return out


@torch.no_grad()
def encode_frames(model, pixels: torch.Tensor) -> torch.Tensor:
    """pixels (B,C,H,W) → emb (B,D)."""
    out = model.encode({"pixels": pixels.unsqueeze(1)})
    return out["emb"][:, 0]


@torch.no_grad()
def encode_episode_frames(
    model,
    ep: EpisodeTraj,
    start: int,
    length: int,
    transform,
    device: torch.device,
) -> torch.Tensor:
    zs = []
    for t in range(start, start + length):
        pix = frame_to_tensor(ep.pixels[t], transform).unsqueeze(0).to(device)
        zs.append(encode_frames(model, pix)[0].cpu())
    return torch.stack(zs, dim=0)


@torch.no_grad()
def imagine_path(
    model,
    z_true: torch.Tensor,
    actions: np.ndarray,
    *,
    device: torch.device,
    history: int = HISTORY,
) -> torch.Tensor:
    """Autoregressive predict from true history embeddings + given actions.

    z_true: (L, D). actions: (L, A) raw env actions (tiled to ACTION_DIM).
    First ``history`` frames of the return are copies of z_true.

    Matches the *loop* of LeWM.rollout, not the extra final predict
    (``n_steps + 1`` in installed ``wm/lewm/lewm.py``). Dump shuffle vs true
    is internally consistent; dump at_h5 is per-frame, not CEM token-horizon.
    """
    L = z_true.size(0)
    HS = history
    z_true = z_true.to(device)
    acts = torch.from_numpy(actions_to_tokens(actions)).to(device)

    emb = z_true[:HS].unsqueeze(0).clone()
    act = acts[:HS].unsqueeze(0).clone()
    out = [z_true[i].cpu() for i in range(HS)]
    n_steps = L - HS
    for t in range(n_steps):
        act_emb = model.action_encoder(act)
        pred = model.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        next_a = acts[HS + t : HS + t + 1].unsqueeze(0)
        act = torch.cat([act, next_a], dim=1)
        out.append(pred[0, 0].cpu())
    return torch.stack(out, dim=0)


@torch.no_grad()
def imagine_closed_loop(
    model,
    z_true: torch.Tensor,
    actions: np.ndarray,
    m: int,
    *,
    device: torch.device,
    history: int = HISTORY,
) -> torch.Tensor:
    """Open-loop chunks of ``m`` predicted steps, then snap to true ``z``.

    First ``history`` frames are always copies of ``z_true``. When
    ``m >= L - history`` this is identical to :func:`imagine_path` (CA0
    ``m=25`` open-loop). ``m=1`` teacher-forces after every predicted frame.
    """
    m = int(m)
    if m < 1:
        raise ValueError(f"re-encode interval m must be >= 1, got {m}")
    L = z_true.size(0)
    HS = history
    n_pred = L - HS
    if n_pred <= 0:
        return z_true[:L].cpu()
    if m >= n_pred:
        return imagine_path(model, z_true, actions, device=device, history=history)

    z_true = z_true.to(device)
    acts = torch.from_numpy(actions_to_tokens(actions)).to(device)
    if acts.size(0) < L:
        pad = torch.zeros(L - acts.size(0), acts.size(-1), device=device, dtype=acts.dtype)
        acts = torch.cat([acts, pad], dim=0)
    elif acts.size(0) > L:
        acts = acts[:L]

    out = [z_true[i].cpu() for i in range(HS)]
    t = HS
    while t < L:
        emb = z_true[t - HS : t].unsqueeze(0).clone()
        act = acts[t - HS : t].unsqueeze(0).clone()
        chunk = min(m, L - t)
        for k in range(chunk):
            act_emb = model.action_encoder(act)
            pred = model.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred], dim=1)
            next_a = acts[t + k : t + k + 1].unsqueeze(0)
            act = torch.cat([act, next_a], dim=1)
            out.append(pred[0, 0].cpu())
        t += chunk
    return torch.stack(out, dim=0)


# PushT contact heuristic (matches GoalPushPolicy.contact_radius).
PUSHT_CONTACT_RADIUS = 45.0
PUSHT_WALL_MARGIN = 40.0
PUSHT_ARENA = 512.0


def contact_events(state: np.ndarray, env: str = "pusht") -> dict[str, np.ndarray]:
    """Per-step contact / wall masks from sim ``state`` (last dim = factors).

    PushT: pusher–block L2 < 45 (GoalPush radius); block within 40 of [0, 512].
    Other envs: empty False masks (Reacher stub).
    """
    st = np.asarray(state, dtype=np.float64)
    if st.ndim == 1:
        st = st.reshape(1, -1)
    leading = st.shape[:-1]
    empty = np.zeros(leading, dtype=bool)
    if env != "pusht" or st.shape[-1] < 4:
        return {"contact": empty, "wall": empty, "any": empty}
    agent = st[..., :2]
    block = st[..., 2:4]
    dist = np.linalg.norm(agent - block, axis=-1)
    contact = dist < PUSHT_CONTACT_RADIUS
    wall = (
        (block[..., 0] < PUSHT_WALL_MARGIN)
        | (block[..., 0] > PUSHT_ARENA - PUSHT_WALL_MARGIN)
        | (block[..., 1] < PUSHT_WALL_MARGIN)
        | (block[..., 1] > PUSHT_ARENA - PUSHT_WALL_MARGIN)
    )
    return {"contact": contact, "wall": wall, "any": contact | wall}


def shuffle_future_actions(
    actions: np.ndarray, history: int, rng: np.random.Generator
) -> np.ndarray:
    out = np.array(actions, copy=True)
    if len(out) > history:
        tail = out[history:]
        rng.shuffle(tail)
        out[history:] = tail
    return out


def per_step_drift(z: torch.Tensor, z_hat: torch.Tensor) -> np.ndarray:
    """‖ẑ_t − z_t‖₂ for t = 0..L-1."""
    return torch.linalg.vector_norm(z_hat - z, ord=2, dim=-1).cpu().numpy()


def summarize_drift(err: np.ndarray, history: int = HISTORY) -> dict[str, float]:
    err = np.asarray(err, dtype=np.float64)
    pred = err[history:] if err.size > history else err
    cem_h = history + CEM_HORIZON - 1
    at_cem = float(err[cem_h]) if err.size > cem_h else float("nan")
    return {
        "mean_all_frames": float(err.mean()) if err.size else float("nan"),
        "mean_predicted_only": float(pred.mean()) if pred.size else float("nan"),
        "at_h5_index": at_cem,
        "end": float(err[-1]) if err.size else float("nan"),
        "history": float(history),
        "cem_horizon": float(CEM_HORIZON),
    }


def save_dump(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in payload.items() if k != "meta"}
    np.savez_compressed(path, **arrays)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(payload["meta"], indent=2))


def load_dump(path: Path) -> dict[str, Any]:
    path = Path(path)
    blob = np.load(path, allow_pickle=True)
    data = {k: blob[k] for k in blob.files}
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        data["meta"] = json.loads(meta_path.read_text())
    else:
        data["meta"] = {}
    return data
