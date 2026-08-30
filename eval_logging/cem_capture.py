"""Capture the last CEM get_cost candidate set for Fig-2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class CemCapture:
    """Wraps model.get_cost; dumps the final-iteration candidates once."""

    def __init__(
        self,
        out_dir: Path,
        *,
        episode: int = 0,
        oracle_actions: np.ndarray | None = None,
    ):
        self.out_dir = Path(out_dir)
        self.episode = int(episode)
        self.oracle_actions = (
            None if oracle_actions is None else np.asarray(oracle_actions, dtype=np.float32)
        )
        self.last_actions: np.ndarray | None = None
        self.last_costs: np.ndarray | None = None
        self.last_info: dict | None = None
        self._orig_get_cost = None
        self._dumped = False
        self._solve_count = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def wrap_model(self, model):
        self._orig_get_cost = model.get_cost

        def hooked(info_dict, action_candidates):
            import torch

            snap = {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in info_dict.items()
            }
            cost = self._orig_get_cost(info_dict, action_candidates)
            try:
                self.last_info = snap
                self.last_actions = action_candidates.detach().float().cpu().numpy()
                self.last_costs = cost.detach().float().cpu().numpy()
            except Exception:
                pass
            return cost

        model.get_cost = hooked
        self.model = model
        return model

    def wrap_solver(self, solver):
        capture = self

        class _Wrapped:
            def __init__(self):
                self._solver = solver

            def configure(self, **kwargs):
                return self._solver.configure(**kwargs)

            def __call__(self, *args, **kwargs):
                return self.solve(*args, **kwargs)

            def solve(self, info_dict, init_action=None):
                outputs = self._solver.solve(info_dict, init_action=init_action)
                capture.on_solve(info_dict, outputs)
                return outputs

            def __getattr__(self, name):
                return getattr(self._solver, name)

        return _Wrapped()

    def on_solve(self, info_dict, outputs) -> None:
        if self._dumped:
            return
        self._solve_count += 1
        if self.last_actions is None or self.last_costs is None:
            return
        acts = np.asarray(self.last_actions)
        costs = np.asarray(self.last_costs)
        if acts.ndim == 4:
            acts = acts[0]
        if costs.ndim == 2:
            costs = costs[0]
        costs = costs.reshape(-1)
        # CEM get_cost is (B, S) or (S,)
        if acts.ndim == 2:
            # (T, A) — one candidate; should not happen for CEM samples
            pass
        selected = int(np.argmin(costs))
        oracle_tok = None
        oracle_cost = None
        if self.oracle_actions is not None and self._orig_get_cost is not None:
            tok = _oracle_to_cem_tokens(self.oracle_actions, acts.shape)
            oracle_tok = tok
            try:
                import torch

                cand = torch.from_numpy(tok)
                while cand.ndim < 4:
                    cand = cand.unsqueeze(0)
                cand = cand.to(next(self.model.parameters()).device)
                info = self.last_info if self.last_info is not None else info_dict
                info = _squeeze_info_one_sample(info)
                oc = self._orig_get_cost(info, cand)
                oracle_cost = float(oc.detach().float().cpu().numpy().reshape(-1)[0])
                errp = self.out_dir / "oracle_cost_error.txt"
                if errp.exists():
                    errp.unlink()
            except Exception as exc:
                oracle_cost = None
                oracle_tok = None
                (self.out_dir / "oracle_cost_error.txt").write_text(str(exc))
        np.savez_compressed(
            self.out_dir / "cem_capture.npz",
            actions=acts.astype(np.float32),
            costs=costs.astype(np.float32),
            selected_idx=np.asarray([selected], dtype=np.int32),
            oracle_action=(
                np.asarray(oracle_tok, dtype=np.float32)
                if oracle_tok is not None
                else np.zeros(0, dtype=np.float32)
            ),
            oracle_cost=np.asarray(
                [oracle_cost if oracle_cost is not None else np.nan], dtype=np.float32
            ),
        )
        meta = {
            "episode": self.episode,
            "solve_index": self._solve_count,
            "n_candidates": int(acts.shape[0]),
            "selected_idx": selected,
            "selected_cost": float(costs[selected]),
            "oracle_cost": oracle_cost,
            "regret": (
                None
                if oracle_cost is None
                else float(oracle_cost - costs[selected])
            ),
        }
        (self.out_dir / "cem_capture.meta.json").write_text(json.dumps(meta, indent=2))
        self._dumped = True


def _squeeze_info_one_sample(info: dict) -> dict:
    """CEM expands tensors to (B, S, ...); oracle eval needs S=1."""
    import torch

    out = {}
    drop = {"predicted_emb", "emb", "action"}
    for k, v in info.items():
        if k in drop:
            continue
        if torch.is_tensor(v) and v.ndim >= 2:
            out[k] = v[:, :1].contiguous() if v.size(1) > 1 else v.contiguous()
        else:
            out[k] = v
    return out


def _oracle_to_cem_tokens(oracle_env: np.ndarray, cem_shape: tuple) -> np.ndarray:
    """Pack env actions into one CEM candidate matching ``cem_shape`` (S, T, A) or (T, A)."""
    from phase_b import pack_action_token

    env_a = np.asarray(oracle_env, dtype=np.float32)
    if env_a.ndim == 1:
        env_a = env_a.reshape(1, -1)
    if len(cem_shape) == 3:
        _s, t, a_dim = cem_shape
    else:
        t, a_dim = cem_shape[-2], cem_shape[-1]
    # Group env steps into T tokens (mean of equal chunks, then tile).
    n = len(env_a)
    chunks = np.array_split(env_a, t) if n >= t else list(env_a) + [env_a[-1]] * (t - n)
    tokens = []
    for ch in chunks[:t]:
        ch = np.asarray(ch)
        mean_a = ch.mean(axis=0) if ch.ndim == 2 else ch
        tokens.append(pack_action_token(mean_a, action_dim=int(a_dim)))
    tok = np.stack(tokens, axis=0)
    return tok
