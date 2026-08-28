"""Shared LeWM eval setup aligned with eval.py (live-env variant).

eval.py fits StandardScaler on an HDF5 dataset; for live eval we fit the same
scalers from short rollouts in the simulator (no dataset download).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def img_transform(img_size: int):
    """Image preprocessing — matches eval.py ``img_transform``."""
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def make_transform(img_size: int) -> dict:
    transform = img_transform(img_size)
    return {"pixels": transform, "goal": transform}


def fit_process(keys_to_cache: list[str], columns: dict[str, np.ndarray]) -> dict:
    """Fit sklearn StandardScalers — same logic as eval.py."""
    process: dict[str, preprocessing.StandardScaler] = {}
    for col in keys_to_cache:
        if col == "pixels":
            continue
        if col not in columns:
            raise KeyError(f"missing column {col!r} for process fitting")
        processor = preprocessing.StandardScaler()
        col_data = np.asarray(columns[col])
        if col_data.ndim == 1:
            col_data = col_data[:, None]
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = processor
    return process


def _append_info_column(
    buffers: dict[str, list[np.ndarray]], key: str, value: Any
) -> None:
    if key not in buffers:
        return
    arr = np.array(value, copy=True)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim >= 2:
        arr = arr.reshape(-1, arr.shape[-1])
    buffers[key].append(arr)


def _sample_actions(world, rng: np.random.Generator) -> np.ndarray:
    space = world.envs.single_action_space
    low = np.broadcast_to(space.low, space.shape)
    high = np.broadcast_to(space.high, space.shape)
    actions = rng.uniform(low, high, size=(world.num_envs, *space.shape))
    return actions.astype(space.dtype, copy=False)


def _pusht_collection_policy(world, seed: int):
    from stable_worldmodel.envs.pusht.expert_policy import WeakPolicy

    policy = WeakPolicy(seed=seed)
    policy.set_env(world.envs)
    return policy


def fit_process_live(
    world,
    keys_to_cache: list[str],
    *,
    num_steps: int = 2000,
    seed: int = 0,
    env_name: str = "",
) -> dict:
    """Estimate eval.py normalizers from live simulator rollouts."""
    buffers = {
        col: []
        for col in keys_to_cache
        if col not in ("pixels",)
    }
    if not buffers:
        return {}

    rng = np.random.default_rng(seed)
    collection_policy = None
    if "PushT" in env_name:
        collection_policy = _pusht_collection_policy(world, seed)

    world.reset(seed=seed)
    for step in range(num_steps):
        for key in buffers:
            if key in world.infos:
                _append_info_column(buffers, key, world.infos[key])

        if collection_policy is not None:
            actions = collection_policy.get_action(world.infos)
        else:
            actions = _sample_actions(world, rng)

        _, _, terminated, truncated, infos = world.envs.step(actions)
        world.infos = infos
        world.terminateds = terminated
        world.truncateds = truncated

        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
        if done.any():
            mask = done
            seeds = [None] * world.num_envs
            base = step + 1
            for rank, env_i in enumerate(np.where(mask)[0]):
                seeds[int(env_i)] = seed + base + rank
            _, infos = world.envs.reset(seed=seeds, mask=mask)
            world.infos = infos
            world.terminateds = np.zeros(world.num_envs, dtype=bool)
            world.truncateds = np.zeros(world.num_envs, dtype=bool)

    columns = {key: np.concatenate(chunks, axis=0) for key, chunks in buffers.items()}
    process = fit_process(keys_to_cache, columns)
    return process


def process_summary(process: dict) -> dict[str, dict[str, list[float]]]:
    summary: dict[str, dict[str, list[float]]] = {}
    for key, scaler in process.items():
        if key.startswith("goal_"):
            continue
        summary[key] = {
            "mean": scaler.mean_.reshape(-1).tolist(),
            "std": np.sqrt(scaler.var_).reshape(-1).tolist(),
            "n_features": int(scaler.n_features_in_),
        }
    return summary


def remap_vit4_to_vit5(state_dict: dict) -> dict:
    """Map transformers 4 ViT keys (HF checkpoints) onto transformers 5."""
    replacements = (
        (".attention.attention.query.", ".attention.q_proj."),
        (".attention.attention.key.", ".attention.k_proj."),
        (".attention.attention.value.", ".attention.v_proj."),
        (".attention.output.dense.", ".attention.o_proj."),
        (".intermediate.dense.", ".mlp.fc1."),
        (".output.dense.", ".mlp.fc2."),
        ("encoder.encoder.layer.", "encoder.layers."),
    )
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


def load_lewm_checkpoint(ckpt_name: str, *, cache_dir: str | Path | None = None):
    """Load LeWM like eval.py, with ViT key remap for HF checkpoints."""
    from hydra.utils import instantiate
    from stable_worldmodel.wm.utils import _resolve_folder
    from stable_worldmodel.data.utils import get_cache_dir

    folder = get_cache_dir(cache_dir, sub_folder="checkpoints") / ckpt_name
    checkpoint_path, config = _resolve_folder(folder)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if any(k.startswith("encoder.encoder.layer.") for k in state):
        state = remap_vit4_to_vit5(state)
    model = instantiate(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if "num_batches_tracked" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch for {ckpt_name}: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    return model


def attach_reach_head(
    model,
    *,
    plan_cost: str = "l2_z",
    phi_weights: str | Path | None = None,
    input_dim: int = 192,
    hidden_dim: int = 256,
    output_dim: int = 64,
    cache_goal_emb: bool = True,
    device: str | None = None,
):
    """Attach lewm-phi ReachabilityHead to a loaded JEPA without rewriting ckpts."""
    from reachability import ReachabilityHead

    if plan_cost not in ("l2_z", "phi_d"):
        raise ValueError(f"plan_cost must be 'l2_z' or 'phi_d', got {plan_cost!r}")

    model.plan_cost = plan_cost
    model.cache_goal_emb = cache_goal_emb
    model.clear_goal_cache()

    if plan_cost == "l2_z" and not phi_weights:
        model.reach = None
        return model

    head = ReachabilityHead(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
    )
    if phi_weights:
        path = Path(phi_weights)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "reach" in state:
            state = state["reach"]
        head.load_state_dict(state)
    model.reach = head
    if device is not None:
        model.reach.to(device)
    model.reach.eval()
    for p in model.reach.parameters():
        p.requires_grad_(False)
    return model


def build_world_model_policy(
    model,
    *,
    process: dict | None,
    img_size: int,
    plan_config: dict[str, Any],
    num_samples: int,
    cem_steps: int,
    topk: int,
    var_scale: float,
    device: str,
    seed: int,
    on_planning_solve=None,
    plan_debugger=None,
    plan_cost: str = "l2_z",
    phi_weights: str | Path | None = None,
    cache_goal_emb: bool = True,
):
    """Build WorldModelPolicy the same way eval.py does."""
    from eval_logging import wrap_solver_timing
    from stable_worldmodel.solver import CEMSolver
    from stable_worldmodel.solver.callbacks import (
        BestCostRecorder,
        EliteCostRecorder,
    )

    attach_reach_head(
        model,
        plan_cost=plan_cost,
        phi_weights=phi_weights,
        cache_goal_emb=cache_goal_emb,
        device=device,
    )

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    if getattr(model, "reach", None) is not None:
        model.reach.to(device)
        model.reach.eval()
    if hasattr(model, "interpolate_pos_encoding"):
        model.interpolate_pos_encoding = True

    cem_callbacks = None
    if plan_debugger is not None:
        cem_callbacks = [EliteCostRecorder(), BestCostRecorder()]

    solver = CEMSolver(
        model=model,
        batch_size=1,
        num_samples=num_samples,
        var_scale=var_scale,
        n_steps=cem_steps,
        topk=topk,
        device=device,
        seed=seed,
        callbacks=cem_callbacks,
    )
    if plan_debugger is not None:
        from eval_logging.plan_debug import wrap_solver_plan_debug

        solver = wrap_solver_plan_debug(
            solver, plan_debugger, on_timing=on_planning_solve
        )
    elif on_planning_solve is not None:
        solver = wrap_solver_timing(solver, on_planning_solve)

    config = swm.PlanConfig(**plan_config)
    return swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process or {},
        transform=make_transform(img_size),
    )
