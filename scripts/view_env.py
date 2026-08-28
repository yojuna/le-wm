#!/usr/bin/env python3
"""Smoke-test the live viewer for any eval_live env (no policy / no CEM).

Examples:
    .venv/bin/python scripts/view_env.py --env reacher
    .venv/bin/python scripts/view_env.py --env pusht --steps 150
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

_shutdown = False


def _sigint(*_):
    global _shutdown
    _shutdown = True


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from eval_live import ENV_REGISTRY
    from live_viewer import ViewerConfig, WorldViewer, configure_gl_backend, supports_viewer
    import stable_worldmodel as swm

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(ENV_REGISTRY), default="reacher")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0, help="sleep multiplier (>1 slower)")
    args = parser.parse_args()

    configure_gl_backend(viewer=True)
    signal.signal(signal.SIGINT, _sigint)

    spec = ENV_REGISTRY[args.env]
    world = swm.World(
        spec.env_name,
        num_envs=1,
        max_episode_steps=spec.max_episode_steps,
        image_shape=(spec.img_size, spec.img_size),
        **spec.world_kwargs,
    )
    if not supports_viewer(world):
        raise SystemExit(f"No viewer backend for {spec.env_name}")

    cfg = ViewerConfig(save_frames=False, img_size=spec.img_size, window_size=512, render_subdir=args.env)
    env = world.envs.envs[0]
    dt = 0.02 * max(args.speed, 0.0)

    with WorldViewer.attach(world, seed=args.seed, config=cfg, env_key=args.env) as viewer:
        print(f"viewer backend: {viewer.backend.name} — close window or press q (pixel) / Ctrl+C")
        for step in range(args.steps):
            if _shutdown:
                break
            if viewer.backend.name == "pixel_window" and not viewer.backend.is_running(viewer.handle):
                break
            if viewer.backend.name == "mujoco_dmc" and not viewer.handle.native.is_running():
                break
            action = env.action_space.sample()
            env.step(action)
            viewer.sync()
            if dt > 0:
                time.sleep(dt)

    world.close()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
