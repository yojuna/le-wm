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
CEM_HORIZON = 5  # EnvSpec / pusht.yaml default; mark on drift plots

PUSHT_FACTORS = (
    "agent_x",
    "agent_y",
    "block_x",
    "block_y",
    "block_angle",
    "agent_vx",
    "agent_vy",
)

DUMP_VERSION = 1


def img_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def pad_action(a: np.ndarray, action_dim: int = ACTION_DIM) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    out = np.zeros(action_dim, dtype=np.float32)
    out[: min(len(a), action_dim)] = a[:action_dim]
    return out


def factor_names_for_env(env: str, state_dim: int) -> tuple[str, ...]:
    if env == "pusht":
        names = list(PUSHT_FACTORS[:state_dim])
        while len(names) < state_dim:
            names.append(f"state_{len(names)}")
        return tuple(names)
    # Reacher: qpos, qvel, finger, target concatenated in dump collector
    generic = []
    for i in range(state_dim):
        generic.append(f"factor_{i}")
    if state_dim >= 2:
        generic[0] = "qpos_0"
        generic[1] = "qpos_1"
    return tuple(generic)


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

    z_true: (L, D). actions: (L, A) raw env actions (padded internally).
    First ``history`` frames of the return are copies of z_true.
    """
    L = z_true.size(0)
    HS = history
    z_true = z_true.to(device)
    acts = torch.from_numpy(
        np.stack([pad_action(a) for a in actions], axis=0)
    ).to(device)

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
