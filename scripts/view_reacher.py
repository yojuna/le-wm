#!/usr/bin/env python3
"""Interactive MuJoCo viewer for the LeWM Reacher env (smoke test).

Two modes:

  swm (default) — real stable-worldmodel ReacherDMControlWrapper (dm_control).
      Requires MUJOCO_GL=glfw (dm_control does not accept glx).

  native — load dm_control's reacher.xml with bare mujoco only (like ws_manipulation).
      Uses MUJOCO_GL=glx. No task logic; physics smoke test only.

Common pattern (both modes):
  launch_passive(model, data) → step → viewer.sync()
  NO swm.World / no offscreen pixel pipeline while viewing.

Usage:

    .venv/bin/python scripts/view_reacher.py
    .venv/bin/python scripts/view_reacher.py --backend native
    .venv/bin/python scripts/view_reacher.py --speed 0.5 --policy zero

Note: `MUJOCO_GL=glx` alone on the shell does nothing unless you export it:
    export MUJOCO_GL=glx   # persists for child processes
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import sys
import time
from pathlib import Path

_shutdown = False


def _handle_sigint(_signum, _frame) -> None:
    global _shutdown
    _shutdown = True


def _dm_control_root() -> Path:
    spec = importlib.util.find_spec("dm_control")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("dm_control is not installed in this venv")
    return Path(spec.submodule_search_locations[0])


def _run_native(args: argparse.Namespace) -> None:
    """Bare mujoco + suite reacher.xml (ws_manipulation-style, glx)."""
    os.environ["MUJOCO_GL"] = "glx"

    import mujoco
    import mujoco.viewer
    import numpy as np

    xml = _dm_control_root() / "suite" / "reacher.xml"
    if not xml.is_file():
        raise FileNotFoundError(f"reacher.xml not found at {xml}")

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.sync()
    print("MuJoCo viewer opened (native reacher.xml)")
    print("  xml:", xml)
    print("  MUJOCO_GL:", os.environ.get("MUJOCO_GL"))

    dt = model.opt.timestep
    if args.speed > 0:
        dt *= args.speed

    rng = np.random.default_rng(args.seed)
    try:
        while viewer.is_running() and not _shutdown:
            if args.policy == "random" and model.nu > 0:
                data.ctrl[:] = rng.uniform(-1.0, 1.0, size=model.nu)
            elif model.nu > 0:
                data.ctrl[:] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            if args.speed > 0:
                time.sleep(dt)
    finally:
        if viewer.is_running():
            viewer.close()
        print("Done.")


def _run_swm(args: argparse.Namespace) -> None:
    """stable-worldmodel ReacherDMControlWrapper (needs dm_control → glfw)."""
    os.environ["MUJOCO_GL"] = "glfw"

    import numpy as np
    import mujoco.viewer
    from stable_worldmodel.envs.dmcontrol.reacher import ReacherDMControlWrapper

    env = ReacherDMControlWrapper(task=args.task, seed=args.seed)
    env.reset(seed=args.seed)

    model = env.env.physics.model.ptr
    data = env.env.physics.data.ptr

    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.sync()
    print("MuJoCo viewer opened (ReacherDMControlWrapper)")
    print("  task:", args.task)
    print("  MUJOCO_GL:", os.environ.get("MUJOCO_GL"))

    dt = float(getattr(env.env, "dt", 0.02)) * env.action_repeat
    if args.speed > 0:
        dt *= args.speed

    try:
        while viewer.is_running() and not _shutdown:
            if args.policy == "random":
                action = env.action_space.sample()
            else:
                action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

            env.step(action)
            viewer.sync()

            if args.speed > 0:
                time.sleep(dt)
    finally:
        if viewer.is_running():
            viewer.close()
        env.close()
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("swm", "native"),
        default="swm",
        help="swm=LeWM reacher wrapper (glfw); native=bare mujoco xml (glx)",
    )
    parser.add_argument(
        "--task",
        default="qpos_match",
        choices=("easy", "hard", "qpos_match"),
        help="Reacher task (swm backend only)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy",
        choices=("random", "zero"),
        default="random",
        help="Action source while the viewer is open",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Wall-clock slowdown (>1 = slower). 0 = uncapped.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)

    if args.backend == "native":
        _run_native(args)
    else:
        _run_swm(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
