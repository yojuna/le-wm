"""Capture and plot CEM planning diagnostics for near-miss debugging."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from eval_logging.extractors import (
    PUSHT_ANGLE_TOL,
    PUSHT_POS_TOL,
    pusht_pose_errors,
    pusht_success,
)


@dataclass
class ReplanRecord:
    episode_idx: int
    replan_idx: int
    env_step: int
    duration_s: float
    final_cost: float | None
    elite_cost_curve: list[float] = field(default_factory=list)
    best_cost_curve: list[float] = field(default_factory=list)
    actions: list[list[float]] = field(default_factory=list)  # (H, A_flat)
    action_norm_mean: float | None = None


@dataclass
class EpisodeTrace:
    episode_idx: int
    seed: int
    pos_errors: list[float] = field(default_factory=list)
    angle_errors: list[float] = field(default_factory=list)
    simultaneous_ok: list[bool] = field(default_factory=list)
    start_pixels: np.ndarray | None = None
    goal_pixels: np.ndarray | None = None
    replans: list[ReplanRecord] = field(default_factory=list)
    success: bool = False
    truncated: bool = False

    @property
    def min_pos(self) -> float | None:
        return min(self.pos_errors) if self.pos_errors else None

    @property
    def min_angle(self) -> float | None:
        return min(self.angle_errors) if self.angle_errors else None

    @property
    def ever_simultaneous(self) -> bool:
        return any(self.simultaneous_ok)

    @property
    def is_near_miss(self) -> bool:
        if self.success or not self.pos_errors:
            return False
        return self.min_pos is not None and self.min_pos < 2.0 * PUSHT_POS_TOL


class PlanDebugger:
    """Collect CEM solve traces and write plots for near-miss episodes."""

    def __init__(
        self,
        output_dir: Path,
        *,
        near_miss_pos: float = 2.0 * PUSHT_POS_TOL,
        enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.near_miss_pos = near_miss_pos
        self.enabled = enabled
        self.episodes: list[EpisodeTrace] = []
        self._current: EpisodeTrace | None = None
        self._replan_counter = 0
        self._env_step = 0
        if enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def begin_episode(self, episode_idx: int, seed: int, infos: dict) -> None:
        if not self.enabled:
            return
        self._current = EpisodeTrace(episode_idx=episode_idx, seed=seed)
        self._replan_counter = 0
        self._env_step = 0
        self._current.start_pixels = _frame(infos.get("pixels"))
        self._current.goal_pixels = _frame(infos.get("goal"))

    def on_step(self, world, env_idx: int = 0) -> None:
        if not self.enabled or self._current is None:
            return
        self._env_step += 1
        state = _vec(world.infos.get("state"), env_idx)
        goal = _vec(world.infos.get("goal_state"), env_idx)
        if state is None or goal is None:
            return
        pos, ang = pusht_pose_errors(goal, state)
        self._current.pos_errors.append(pos)
        self._current.angle_errors.append(ang)
        self._current.simultaneous_ok.append(pusht_success(goal, state))

    def note_solve(
        self,
        duration_s: float,
        outputs: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._current is None:
            return
        elite_curve, best_curve = _extract_cost_curves(outputs)
        actions = _extract_actions(outputs)
        final_cost = None
        if outputs and outputs.get("costs"):
            final_cost = float(np.mean(outputs["costs"]))
        action_norm = None
        if actions:
            arr = np.asarray(actions, dtype=np.float64)
            action_norm = float(np.linalg.norm(arr, axis=-1).mean())

        rec = ReplanRecord(
            episode_idx=self._current.episode_idx,
            replan_idx=self._replan_counter,
            env_step=self._env_step,
            duration_s=duration_s,
            final_cost=final_cost,
            elite_cost_curve=elite_curve,
            best_cost_curve=best_curve,
            actions=actions,
            action_norm_mean=action_norm,
        )
        self._current.replans.append(rec)
        self._replan_counter += 1

        # Always dump per-replan lightweight JSON + cost plot
        ep_dir = self.output_dir / f"ep{self._current.episode_idx:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_replan_json(ep_dir / f"replan_{rec.replan_idx:02d}.json", rec)
        if elite_curve or best_curve:
            _plot_cost_curve(
                ep_dir / f"replan_{rec.replan_idx:02d}_costs.png",
                elite_curve,
                best_curve,
                title=f"ep{self._current.episode_idx} replan{rec.replan_idx}",
            )
        if actions:
            _plot_actions(
                ep_dir / f"replan_{rec.replan_idx:02d}_actions.png",
                actions,
                title=f"ep{self._current.episode_idx} replan{rec.replan_idx} plan",
            )

    def end_episode(self, success: bool, truncated: bool) -> None:
        if not self.enabled or self._current is None:
            return
        self._current.success = success
        self._current.truncated = truncated
        ep = self._current
        self.episodes.append(ep)
        ep_dir = self.output_dir / f"ep{ep.episode_idx:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        _plot_pose_trace(
            ep_dir / "pose_trace.png",
            ep,
            near_miss_pos=self.near_miss_pos,
        )
        if ep.start_pixels is not None or ep.goal_pixels is not None:
            _plot_start_goal(
                ep_dir / "start_goal.png",
                ep.start_pixels,
                ep.goal_pixels,
                title=f"ep{ep.episode_idx} seed={ep.seed}",
            )

        summary = {
            "episode_idx": ep.episode_idx,
            "seed": ep.seed,
            "success": ep.success,
            "truncated": ep.truncated,
            "min_pos_error": ep.min_pos,
            "min_angle_error": ep.min_angle,
            "ever_simultaneous_success": ep.ever_simultaneous,
            "near_miss": ep.is_near_miss,
            "n_replans": len(ep.replans),
            "final_pos_error": ep.pos_errors[-1] if ep.pos_errors else None,
            "final_angle_error": ep.angle_errors[-1] if ep.angle_errors else None,
            "pos_tol": PUSHT_POS_TOL,
            "angle_tol": float(PUSHT_ANGLE_TOL),
        }
        (ep_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        if ep.is_near_miss or ep.ever_simultaneous:
            _plot_near_miss_panel(ep_dir / "near_miss_panel.png", ep)
            tag = "SIMULTANEOUS_OK_BUT_NO_TERM" if ep.ever_simultaneous and not ep.success else "NEAR_MISS"
            print(
                f"  [plan-debug] {tag} ep={ep.episode_idx} "
                f"min_pos={ep.min_pos:.2f} min_ang={ep.min_angle:.3f} "
                f"ever_simul={ep.ever_simultaneous} → {ep_dir}"
            )

        self._current = None

    def finalize(self) -> Path | None:
        if not self.enabled:
            return None
        overview = {
            "n_episodes": len(self.episodes),
            "n_near_miss": sum(1 for e in self.episodes if e.is_near_miss),
            "n_ever_simultaneous": sum(1 for e in self.episodes if e.ever_simultaneous),
            "n_success": sum(1 for e in self.episodes if e.success),
            "episodes": [
                {
                    "episode_idx": e.episode_idx,
                    "seed": e.seed,
                    "success": e.success,
                    "min_pos": e.min_pos,
                    "min_angle": e.min_angle,
                    "near_miss": e.is_near_miss,
                    "ever_simultaneous": e.ever_simultaneous,
                }
                for e in self.episodes
            ],
        }
        path = self.output_dir / "overview.json"
        path.write_text(json.dumps(overview, indent=2) + "\n")
        print(f"plan-debug overview: {path}")
        return path


def wrap_solver_plan_debug(solver: Any, debugger: PlanDebugger, on_timing=None):
    """Wrap solver to record timing + full CEM outputs for PlanDebugger."""

    class _Wrapped:
        def __init__(self):
            self._solver = solver

        def configure(self, **kwargs):
            return self._solver.configure(**kwargs)

        def __call__(self, *args, **kwargs):
            return self.solve(*args, **kwargs)

        def solve(self, info_dict, init_action=None):
            import time

            t0 = time.perf_counter()
            outputs = self._solver.solve(info_dict, init_action=init_action)
            dt = time.perf_counter() - t0
            if on_timing is not None:
                on_timing(dt)
            debugger.note_solve(dt, outputs)
            return outputs

        def __getattr__(self, name):
            return getattr(self._solver, name)

    return _Wrapped()


# ---- helpers -----------------------------------------------------------------


def _frame(value) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim >= 4:
        arr = arr[0]
    if arr.ndim >= 4:
        arr = arr[-1]
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return np.asarray(arr, copy=True)


def _vec(value, env_idx: int = 0) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value[env_idx]).reshape(-1)
    return arr.astype(np.float64, copy=False)


def _extract_cost_curves(outputs: dict | None) -> tuple[list[float], list[float]]:
    if not outputs:
        return [], []
    cbs = outputs.get("callbacks") or {}
    elite, best = [], []
    if "EliteCostRecorder" in cbs:
        hist = cbs["EliteCostRecorder"]
        # history: list[batch][step] -> dict with mean/min/max
        flat = _flatten_cb_history(hist)
        for item in flat:
            if isinstance(item, dict) and "mean" in item:
                elite.append(float(item["mean"]))
            elif isinstance(item, (int, float)):
                elite.append(float(item))
    if "BestCostRecorder" in cbs:
        hist = cbs["BestCostRecorder"]
        flat = _flatten_cb_history(hist)
        for item in flat:
            if isinstance(item, (int, float)):
                best.append(float(item))
            elif isinstance(item, dict) and "min" in item:
                best.append(float(item["min"]))
    return elite, best


def _flatten_cb_history(hist) -> list:
    out = []
    if not isinstance(hist, list):
        return out
    for batch in hist:
        if isinstance(batch, list):
            out.extend(batch)
        else:
            out.append(batch)
    return out


def _extract_actions(outputs: dict | None) -> list[list[float]]:
    if not outputs or "actions" not in outputs:
        return []
    act = outputs["actions"]
    if hasattr(act, "detach"):
        act = act.detach().cpu().numpy()
    act = np.asarray(act)
    # (B, H, A) → take env 0
    if act.ndim == 3:
        act = act[0]
    return act.astype(np.float64).tolist()


def _write_replan_json(path: Path, rec: ReplanRecord) -> None:
    payload = {
        "episode_idx": rec.episode_idx,
        "replan_idx": rec.replan_idx,
        "env_step": rec.env_step,
        "duration_s": rec.duration_s,
        "final_cost": rec.final_cost,
        "elite_cost_curve": rec.elite_cost_curve,
        "best_cost_curve": rec.best_cost_curve,
        "action_norm_mean": rec.action_norm_mean,
        "actions_shape": [len(rec.actions), len(rec.actions[0]) if rec.actions else 0],
        "actions": rec.actions,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _plot_cost_curve(path, elite, best, title=""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    if elite:
        ax.plot(elite, label="elite mean cost", color="C0")
    if best:
        ax.plot(best, label="best sample cost", color="C1", alpha=0.8)
    ax.set_xlabel("CEM iteration")
    ax.set_ylabel("latent cost")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_actions(path, actions, title=""):
    import matplotlib.pyplot as plt

    arr = np.asarray(actions, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for d in range(min(arr.shape[1], 4)):
        ax.plot(arr[:, d], label=f"a[{d}]")
    if arr.shape[1] > 4:
        # blocked actions: show L2 per horizon step
        ax.plot(np.linalg.norm(arr, axis=1), label="||a||", color="k", ls="--")
    ax.set_xlabel("plan step (blocked)")
    ax.set_ylabel("normalized action")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_pose_trace(path, ep: EpisodeTrace, near_miss_pos: float):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    t = np.arange(len(ep.pos_errors))
    axes[0].plot(t, ep.pos_errors, color="C0", label="pos error")
    axes[0].axhline(PUSHT_POS_TOL, color="C3", ls="--", label=f"tol={PUSHT_POS_TOL}")
    axes[0].axhline(near_miss_pos, color="C1", ls=":", label=f"near={near_miss_pos}")
    axes[0].set_ylabel("pos L2")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(
        f"ep{ep.episode_idx} seed={ep.seed} success={ep.success} "
        f"simul={ep.ever_simultaneous}"
    )

    axes[1].plot(t, ep.angle_errors, color="C2", label="angle error")
    axes[1].axhline(PUSHT_ANGLE_TOL, color="C3", ls="--", label="tol=π/9")
    axes[1].set_xlabel("env step")
    axes[1].set_ylabel("|Δθ|")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_start_goal(path, start, goal, title=""):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, img, name in (
        (axes[0], start, "start / pixels"),
        (axes[1], goal, "goal"),
    ):
        if img is None:
            ax.set_title(f"{name} (missing)")
            ax.axis("off")
            continue
        frame = np.asarray(img)
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        ax.imshow(frame)
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_near_miss_panel(path, ep: EpisodeTrace):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    for ax, img, name in (
        (ax0, ep.start_pixels, "start"),
        (ax1, ep.goal_pixels, "goal"),
    ):
        if img is not None:
            frame = np.asarray(img)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8) if frame.max() > 1 else (frame * 255).astype(np.uint8)
            ax.imshow(frame)
        ax.set_title(name)
        ax.axis("off")

    ax2 = fig.add_subplot(gs[1, :])
    t = np.arange(len(ep.pos_errors))
    ax2.plot(t, ep.pos_errors, label="pos")
    ax2.plot(t, np.asarray(ep.angle_errors) * (PUSHT_POS_TOL / PUSHT_ANGLE_TOL),
             label="angle (scaled)", alpha=0.8)
    ax2.axhline(PUSHT_POS_TOL, color="C3", ls="--", label="pos tol")
    simul_idx = [i for i, ok in enumerate(ep.simultaneous_ok) if ok]
    if simul_idx:
        ax2.scatter(simul_idx, [ep.pos_errors[i] for i in simul_idx],
                    color="C2", s=40, zorder=5, label="simultaneous OK")
    ax2.set_xlabel("step")
    ax2.set_title(
        f"NEAR MISS ep{ep.episode_idx}: min_pos={ep.min_pos:.2f} "
        f"min_ang={ep.min_angle:.3f} ever_simul={ep.ever_simultaneous}"
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
