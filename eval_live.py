"""Live-env LeWM planning baseline (no HDF5 dataset).

Runs CEM MPC in the simulator. For PushT, default protocol ``online_offset``
matches paper eval.py: WeakPolicy trajectories → (start, start+25) pairs →
eval_budget=50 with goal reinjection. Optional ``live_reset`` keeps the harder
open-loop random-goal stress test.

Examples:
  python eval_live.py --download
  python eval_live.py --env pusht --episodes 4
  python eval_live.py --env pusht --protocol live_reset --episodes 4
  python eval_live.py --env reacher --episodes 4 --num-envs 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT.parent / "stablewm"
os.environ.setdefault("STABLEWM_HOME", str(DEFAULT_CACHE))

HF_BASE = "https://huggingface.co"
MIN_WEIGHT_BYTES = 50_000_000


@dataclass
class EnvSpec:
    """One live-env experiment. Copy this to add Cube, TwoRoom, etc."""

    env_name: str
    hf_repo: str
    ckpt_dir: str
    world_kwargs: dict = field(default_factory=dict)
    keys_to_cache: list[str] = field(default_factory=list)
    action_block: int = 5
    horizon: int = 5
    receding_horizon: int = 5
    max_episode_steps: int = 200
    goal_offset_steps: int = 25
    eval_budget: int = 50
    default_protocol: str = "live_reset"
    img_size: int = 224
    notes: str = ""


ENV_REGISTRY: dict[str, EnvSpec] = {
    "pusht": EnvSpec(
        env_name="swm/PushT-v1",
        hf_repo="quentinll/lewm-pusht",
        ckpt_dir="hf_pusht",
        world_kwargs={},
        keys_to_cache=["action", "proprio", "state"],
        max_episode_steps=100,  # 2 * eval_budget, matches eval.py
        goal_offset_steps=25,
        eval_budget=50,
        default_protocol="online_offset",
        notes=(
            "Pymunk PushT. online_offset: kinematic start→goal pairs "
            "(goal_offset=25, budget=50) + WeakPolicy action scaler. "
            "Success = pose match to goal_state, not green-T coverage."
        ),
    ),
    "reacher": EnvSpec(
        env_name="swm/ReacherDMControl-v0",
        hf_repo="quentinll/lewm-reacher",
        ckpt_dir="hf_reacher",
        world_kwargs={"task": "qpos_match"},
        keys_to_cache=["action"],
        max_episode_steps=100,
        goal_offset_steps=25,
        eval_budget=50,
        default_protocol="live_reset",
        notes="MuJoCo DMC 2-joint arm. qpos_match matches the LeWM paper setup.",
    ),
}


def cache_dir() -> Path:
    return Path(os.environ["STABLEWM_HOME"])


def ckpt_folder(spec: EnvSpec) -> Path:
    return cache_dir() / "checkpoints" / spec.ckpt_dir


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    subprocess.check_call(
        [
            "curl",
            "-fL",
            "--retry",
            "8",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            "-C",
            "-",
            "-o",
            str(dest),
            url,
        ]
    )


def _complete_enough(path: Path, name: str) -> bool:
    if not path.exists():
        return False
    size = path.stat().st_size
    if name == "weights.pt":
        return size >= MIN_WEIGHT_BYTES
    return size > 0


def download_spec(spec: EnvSpec, force: bool = False) -> Path:
    folder = ckpt_folder(spec)
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "weights.pt"):
        dest = folder / name
        if _complete_enough(dest, name) and not force:
            print(f"exists {dest} ({dest.stat().st_size} bytes)")
            continue
        url = f"{HF_BASE}/{spec.hf_repo}/resolve/main/{name}"
        download_file(url, dest)
        print(f"saved {dest} ({dest.stat().st_size} bytes)")
    return folder


def load_lewm(spec: EnvSpec):
    from eval_setup import load_lewm_checkpoint

    return load_lewm_checkpoint(spec.ckpt_dir)


def build_policy(spec: EnvSpec, args, *, process=None, on_planning_solve=None, plan_debugger=None):
    import torch
    from eval_setup import build_world_model_policy

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to cpu")
        device = "cpu"

    weights = ckpt_folder(spec) / "weights.pt"
    if not _complete_enough(weights, "weights.pt"):
        download_spec(spec)
    model = load_lewm(spec)

    plan_config = {
        "horizon": spec.horizon,
        "receding_horizon": spec.receding_horizon,
        "action_block": spec.action_block,
        "warm_start": True,
    }
    phi_weights = getattr(args, "phi_weights", "") or None
    return build_world_model_policy(
        model,
        process=process,
        img_size=spec.img_size,
        plan_config=plan_config,
        num_samples=args.num_samples,
        cem_steps=args.cem_steps,
        topk=args.topk,
        var_scale=args.var_scale,
        device=device,
        seed=args.seed,
        on_planning_solve=on_planning_solve,
        plan_debugger=plan_debugger,
        plan_cost=getattr(args, "plan_cost", "l2_z"),
        phi_weights=phi_weights,
        cache_goal_emb=not getattr(args, "no_cache_goal_emb", False),
    )


def install_reacher_goal_hook(world, *, img_size: int, seed: int) -> None:
    """Render goal pixels for Reacher qpos_match (env has no built-in goal image)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    goals: list[np.ndarray | None] = [None] * world.num_envs

    def render_goal(env_idx: int) -> np.ndarray:
        env = world.envs.envs[env_idx].unwrapped
        qpos = np.copy(env.env.physics.data.qpos)
        qvel = np.copy(env.env.physics.data.qvel)
        target_qpos = np.clip(qpos + rng.uniform(-1.0, 1.0, size=qpos.shape), -np.pi, np.pi)
        env.set_target_qpos(target_qpos)
        env.set_state(target_qpos, np.zeros_like(qvel))
        goal = env.render(width=img_size, height=img_size)
        env.set_state(qpos, qvel)
        return goal

    def refresh_goals(mask: np.ndarray | None = None) -> None:
        if mask is None:
            indices = range(world.num_envs)
        else:
            indices = np.where(mask)[0]
        for i in indices:
            goals[int(i)] = render_goal(int(i))

    def inject_goals() -> None:
        world.infos["goal"] = np.stack(goals, axis=0)[:, None, ...]

    refresh_goals()
    inject_goals()

    orig_get_actions = world._get_actions

    def patched_get_actions():
        inject_goals()
        return orig_get_actions()

    world._get_actions = patched_get_actions

    orig_envs_reset = world.envs.reset

    def patched_envs_reset(*args, **kwargs):
        infos = orig_envs_reset(*args, **kwargs)
        mask = kwargs.get("mask")
        refresh_goals(mask if mask is not None else np.ones(world.num_envs, dtype=bool))
        inject_goals()
        return infos

    world.envs.reset = patched_envs_reset

    orig_world_reset = world.reset

    def patched_world_reset(*args, **kwargs):
        out = orig_world_reset(*args, **kwargs)
        refresh_goals()
        inject_goals()
        return out

    world.reset = patched_world_reset


def _resolve_protocol(spec: EnvSpec, args) -> str:
    if args.protocol:
        return args.protocol
    return spec.default_protocol


def _run_meta(spec: EnvSpec, args, *, protocol: str, process_summary=None, extra=None) -> dict:
    meta = {
        "env_name": spec.env_name,
        "hf_repo": spec.hf_repo,
        "checkpoint": str(ckpt_folder(spec)),
        "episodes": args.episodes,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "protocol": protocol,
        "normalization": {
            "source": (
                "weak_policy_bank"
                if protocol == "online_offset"
                else "live_rollout"
            ),
            "stats_steps": args.stats_steps,
            "keys": spec.keys_to_cache,
            "scalers": process_summary or {},
        },
        "plan_config": {
            "horizon": spec.horizon,
            "receding_horizon": spec.receding_horizon,
            "action_block": spec.action_block,
            "steps_per_replan": spec.receding_horizon * spec.action_block,
            "max_episode_steps": spec.max_episode_steps,
            "goal_offset_steps": spec.goal_offset_steps,
            "eval_budget": spec.eval_budget,
        },
        "cem": {
            "num_samples": args.num_samples,
            "n_steps": args.cem_steps,
            "topk": args.topk,
            "var_scale": args.var_scale,
        },
        "device": args.device,
        "notes": spec.notes,
    }
    if extra:
        meta.update(extra)
    return meta


def _log_config(args):
    from eval_logging import EvalLogConfig

    return EvalLogConfig(
        output_dir=Path(args.log_dir),
        run_name=args.run_name,
        save_json=not args.no_save_metrics,
        save_episode_csv=not args.no_save_metrics,
        quiet=args.quiet_logs,
    )


def run_eval(spec: EnvSpec, args):
    import stable_worldmodel as swm
    from eval_logging import (
        EvalRunLogger,
        evaluate_goal_offset,
        evaluate_logged,
        install_mpc_buffer_fix,
    )
    from eval_logging.pairs import (
        collect_trajectory_bank,
        fit_process_from_bank,
        sample_eval_pairs,
    )
    from eval_setup import fit_process_live, process_summary
    from live_viewer import (
        ViewerConfig,
        WorldViewer,
        evaluate_with_viewer,
        supports_viewer,
    )

    protocol = _resolve_protocol(spec, args)
    if protocol == "online_offset" and args.num_envs != 1:
        print("warning: online_offset requires num_envs=1; overriding")
        args.num_envs = 1
    if protocol == "online_offset" and args.viewer:
        print("warning: --viewer not supported with online_offset; disabling viewer")
        args.viewer = False

    if args.viewer:
        if args.num_envs != 1:
            print("warning: --viewer requires num_envs=1; overriding")
            args.num_envs = 1
        if args.video:
            print("warning: --viewer disables file video export (use --no-video)")
            args.video = False

    if args.topk > args.num_samples:
        print(
            f"warning: topk={args.topk} > num_samples={args.num_samples}; "
            f"clamping topk to {args.num_samples}"
        )
        args.topk = args.num_samples

    log_config = _log_config(args)
    world_kwargs = dict(
        env_name=spec.env_name,
        num_envs=args.num_envs,
        max_episode_steps=spec.max_episode_steps,
        image_shape=(spec.img_size, spec.img_size),
        **spec.world_kwargs,
    )
    print(f"creating world {world_kwargs}")
    world = swm.World(**world_kwargs)

    if spec.env_name == "swm/ReacherDMControl-v0":
        install_reacher_goal_hook(world, img_size=spec.img_size, seed=args.seed)

    process = {}
    pairs = None
    bank_meta = {}

    if protocol == "online_offset":
        collect_steps = max(args.stats_steps, args.episodes * (spec.goal_offset_steps + 5) * 4)
        n_kin_eps = max(48, args.episodes * 4)
        print(
            f"collecting {args.collector} bank for "
            f"scalers + {args.episodes} goal_offset={spec.goal_offset_steps} pairs"
        )
        bank = collect_trajectory_bank(
            world,
            num_steps=collect_steps,
            seed=args.seed,
            env_name=spec.env_name,
            min_episode_len=spec.goal_offset_steps + 2,
            collector=args.collector,
            num_episodes=n_kin_eps if args.collector == "kinematic" else None,
            kinematic_horizon=80,
        )
        from eval_logging.pairs import bank_success_report

        report = bank_success_report(bank)
        print(
            f"  bank: {report['episodes']} episodes, {report['steps']} steps, "
            f"collector_success={report['success_episodes']} "
            f"({report['success_rate_pct']:.1f}%)"
        )
        if spec.keys_to_cache and not args.no_normalize:
            process = fit_process_from_bank(bank, spec.keys_to_cache)
            # Kinematic finite-difference actions have tiny variance and would
            # collapse CEM after inverse_transform. Refit action from WeakPolicy.
            if args.collector == "kinematic" and "action" in spec.keys_to_cache:
                print("  refitting action StandardScaler from WeakPolicy rollouts")
                from eval_logging.pairs import collect_trajectory_bank as _collect

                weak_bank = _collect(
                    world,
                    num_steps=max(1500, args.stats_steps // 2),
                    seed=args.seed + 10_000,
                    env_name=spec.env_name,
                    min_episode_len=2,
                    collector="weak",
                )
                weak_process = fit_process_from_bank(weak_bank, ["action"])
                process["action"] = weak_process["action"]
            summary = process_summary(process)
            for key, stats in summary.items():
                print(f"  {key}: mean={stats['mean']} std={stats['std']}")
        elif args.no_normalize:
            print("warning: running without action/state normalization")

        pairs = sample_eval_pairs(
            bank,
            num_eval=args.episodes,
            goal_offset=spec.goal_offset_steps,
            seed=args.seed,
            prefer_success=True,
            min_pos_delta=args.min_pos_delta,
            max_pos_delta=args.max_pos_delta,
            mode=args.pair_mode,
        )
        n_ok_pairs = sum(1 for p in pairs if p.from_success_ep)
        mean_prog = sum(p.pos_progress for p in pairs) / max(len(pairs), 1)
        print(
            f"  pairs: {len(pairs)} mode={args.pair_mode} "
            f"(from_success_ep={n_ok_pairs}, mean_pos_progress={mean_prog:.1f})"
        )
        bank_meta = {
            "pair_source": f"{args.collector}_policy_live",
            "collector": args.collector,
            "pair_mode": args.pair_mode,
            "goal_offset_steps": spec.goal_offset_steps,
            "eval_budget": spec.eval_budget,
            "bank_episodes": report["episodes"],
            "bank_steps": report["steps"],
            "bank_success_episodes": report["success_episodes"],
            "bank_success_rate_pct": report["success_rate_pct"],
            "collect_steps": collect_steps,
            "pairs_from_success_ep": n_ok_pairs,
            "pairs_mean_pos_progress": mean_prog,
            "min_pos_delta": args.min_pos_delta,
            "max_pos_delta": args.max_pos_delta,
        }
    else:
        if spec.keys_to_cache and not args.no_normalize:
            print(
                f"fitting StandardScaler on {args.stats_steps} live steps "
                f"for {spec.keys_to_cache}"
            )
            process = fit_process_live(
                world,
                spec.keys_to_cache,
                num_steps=args.stats_steps,
                seed=args.seed,
                env_name=spec.env_name,
            )
            summary = process_summary(process)
            for key, stats in summary.items():
                print(f"  {key}: mean={stats['mean']} std={stats['std']}")
        elif args.no_normalize:
            print("warning: running without action/state normalization")

    run_meta = _run_meta(
        spec,
        args,
        protocol=protocol,
        process_summary=process_summary(process) if process else None,
        extra=bank_meta or None,
    )
    # Logger meta.protocol used in prints; keep consistent
    eval_logger = EvalRunLogger(
        env_key=args.env,
        config=log_config,
        meta=run_meta,
    )

    plan_debugger = None
    if getattr(args, "plan_debug", False):
        from eval_logging.plan_debug import PlanDebugger

        debug_dir = Path(args.log_dir) / args.env / (
            args.run_name or f"{args.env}_seed{args.seed}"
        ) / "plan_debug"
        plan_debugger = PlanDebugger(debug_dir, enabled=True)
        print(f"plan-debug enabled → {debug_dir}")

    policy = build_policy(
        spec,
        args,
        process=process,
        on_planning_solve=eval_logger.note_planning,
        plan_debugger=plan_debugger,
    )
    install_mpc_buffer_fix(world)
    world.set_policy(policy)

    if not supports_viewer(world) and args.viewer:
        world.close()
        raise SystemExit(f"No viewer backend for {spec.env_name}")

    viewer_session = None
    viewer_backend = None
    viewer_partial = None
    viewer_cfg = None
    metrics = None
    try:
        if args.viewer:
            viewer_cfg = ViewerConfig(
                hold_after_eval=args.viewer_hold,
                save_frames=not args.no_save_viewer_frames,
                img_size=spec.img_size,
                window_size=512 if args.env == "pusht" else None,
                render_subdir=args.env,
            )
            viewer_backend, viewer_partial, viewer_cfg = WorldViewer.prepare(
                world, config=viewer_cfg, env_key=args.env
            )

        video_dir = None
        if args.video:
            video_dir = cache_dir() / "videos" / args.env
            video_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"evaluate env={args.env} protocol={protocol} "
            f"episodes={args.episodes} num_envs={args.num_envs} "
            f"viewer={args.viewer} ({spec.notes})"
        )

        if protocol == "online_offset":
            metrics = evaluate_goal_offset(
                world,
                eval_logger,
                pairs,
                eval_budget=spec.eval_budget,
                video_dir=video_dir,
                plan_debugger=plan_debugger,
            )
        elif args.viewer:
            viewer_session = WorldViewer.open_prepared(
                world,
                viewer_backend,
                viewer_partial,
                seed=args.seed,
                config=viewer_cfg,
            )
            print(
                f"viewer opened ({viewer_session.backend.name}) — "
                "MuJoCo 3D for dm_control envs, 2D window for pixel envs like PushT"
            )
            metrics = evaluate_with_viewer(
                world,
                viewer_session,
                eval_logger,
                episodes=args.episodes,
                seed=args.seed,
                render_dir=viewer_cfg.render_dir(args.env),
            )
            if args.viewer_hold:
                viewer_session.hold_until_closed()
        else:
            metrics = evaluate_logged(
                world,
                eval_logger,
                episodes=args.episodes,
                seed=args.seed,
                video_dir=video_dir,
            )
    except KeyboardInterrupt:
        print("interrupted — saving partial metrics")
        metrics = eval_logger.finalize(interrupted=True)
    finally:
        if viewer_session is not None:
            viewer_session.close()
        world.close()

    return metrics or {}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=sorted(ENV_REGISTRY), default="pusht")
    p.add_argument("--all", action="store_true", help="run every env in the registry")
    p.add_argument(
        "--download",
        action="store_true",
        help="download checkpoints and exit (no planning)",
    )
    p.add_argument(
        "--protocol",
        choices=["online_offset", "live_reset"],
        default="",
        help="eval protocol (default: per-env; PushT → online_offset)",
    )
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-samples", type=int, default=300)
    p.add_argument("--cem-steps", type=int, default=30)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--var-scale", type=float, default=1.0)
    p.add_argument(
        "--stats-steps",
        type=int,
        default=2000,
        help="live rollout steps for scalers / online pair bank",
    )
    p.add_argument(
        "--collector",
        choices=["kinematic", "goal", "weak"],
        default="kinematic",
        help="PushT collection for online_offset (default: kinematic start→goal)",
    )
    p.add_argument(
        "--pair-mode",
        choices=["offset", "short_horizon", "finish"],
        default="offset",
        help=(
            "how to sample start/goal: offset=paper-style t→t+offset; "
            "short_horizon=small reachable hops; finish=start near goal"
        ),
    )
    p.add_argument(
        "--min-pos-delta",
        type=float,
        default=20.0,
        help="min start→goal pose delta when sampling online_offset pairs",
    )
    p.add_argument(
        "--max-pos-delta",
        type=float,
        default=55.0,
        help="max start→goal pose delta (keeps pairs short-horizon like paper)",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="skip StandardScaler (debug only; not like eval.py)",
    )
    p.add_argument(
        "--viewer",
        action="store_true",
        help="open interactive viewer (live_reset only; num_envs=1)",
    )
    p.add_argument(
        "--viewer-hold",
        action="store_true",
        default=True,
        help="keep viewer open after eval until the window is closed (default: on)",
    )
    p.add_argument(
        "--no-viewer-hold",
        action="store_false",
        dest="viewer_hold",
        help="close the viewer immediately when evaluation finishes",
    )
    p.add_argument(
        "--no-save-viewer-frames",
        action="store_true",
        help="with --viewer, do not write per-step pixel PNGs under ws_lewm/renders/",
    )
    p.add_argument("--video", action="store_true", default=True)
    p.add_argument("--no-video", action="store_false", dest="video")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "eval_results",
        help="directory for metrics.json and episodes.csv",
    )
    p.add_argument(
        "--run-name",
        default="",
        help="subfolder name under log-dir/<env>/ (default: <env>_seed<seed>)",
    )
    p.add_argument(
        "--no-save-metrics",
        action="store_true",
        help="skip writing metrics.json / episodes.csv",
    )
    p.add_argument(
        "--quiet-logs",
        action="store_true",
        help="suppress per-episode console lines and summary",
    )
    p.add_argument(
        "--plan-debug",
        action="store_true",
        help="log CEM cost curves, action plans, start/goal images, near-miss panels",
    )
    p.add_argument(
        "--plan-cost",
        choices=["l2_z", "phi_d"],
        default="l2_z",
        help="CEM cost: legacy L2 in z, or Euclidean in φ (lewm-phi)",
    )
    p.add_argument(
        "--phi-weights",
        default="",
        help="path to reach.pt (empty + plan-cost=phi_d → random φ / E4)",
    )
    p.add_argument(
        "--no-cache-goal-emb",
        action="store_true",
        help="disable C1 goal embedding cache (re-encode goal every CEM call)",
    )
    return p.parse_args(argv)


def main(argv=None):
    from live_viewer import configure_gl_backend

    args = parse_args(argv)
    configure_gl_backend(viewer=args.viewer and not args.download)

    names = list(ENV_REGISTRY) if args.all else [args.env]
    if args.download:
        for name in names:
            download_spec(ENV_REGISTRY[name])
        return

    for name in names:
        args.env = name
        print(f"\n=== {name} ===")
        run_eval(ENV_REGISTRY[name], args)


if __name__ == "__main__":
    main()
