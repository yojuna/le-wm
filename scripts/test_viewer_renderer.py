#!/usr/bin/env python3
"""Smoke test: live_viewer EGL+GLFW path for dm_control envs.

Usage:
    .venv/bin/python scripts/test_viewer_renderer.py
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _sigint)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numpy as np
    from PIL import Image
    import stable_worldmodel as swm
    from live_viewer import ViewerConfig, WorldViewer, configure_gl_backend, renders_dir

    configure_gl_backend(viewer=True)

    world = swm.World(
        "swm/ReacherDMControl-v0",
        num_envs=1,
        max_episode_steps=50,
        image_shape=(224, 224),
        task="qpos_match",
    )
    cfg = ViewerConfig(save_frames=False, img_size=224, render_subdir="reacher")
    out_path = args.out or (renders_dir() / "viewer_smoke.png")

    last_rgb = None
    with WorldViewer.attach(world, seed=0, config=cfg, env_key="reacher") as viewer:
        print("viewer open — arm should be visible and moving")
        for _ in range(args.steps):
            if _shutdown:
                break
            if viewer.backend.name == "mujoco_dmc" and not viewer.handle.native.is_running():
                break
            world.envs.envs[0].step(world.envs.envs[0].action_space.sample())
            viewer.sync()
            last_rgb = viewer.render_pixels()
            time.sleep(0.03)

    if last_rgb is not None:
        arr = np.asarray(last_rgb)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(out_path)
        print(f"saved {out_path} shape={arr.shape} min={arr.min()} max={arr.max()}")

    world.close()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
