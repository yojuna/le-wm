"""Viewer configuration and output paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ws_lewm/renders/
PKG_ROOT = Path(__file__).resolve().parents[1]
RENDERS_DIR = PKG_ROOT.parent / "renders"


@dataclass
class ViewerConfig:
    """Options for interactive eval viewing."""

    hold_after_eval: bool = True
    save_frames: bool = True
    img_size: int = 224
    window_size: int | None = None  # display size; defaults to img_size
    render_subdir: str | None = None  # defaults to short env key, e.g. "reacher"

    def render_dir(self, env_key: str) -> Path | None:
        if not self.save_frames:
            return None
        sub = self.render_subdir or env_key
        path = RENDERS_DIR / sub
        path.mkdir(parents=True, exist_ok=True)
        return path


def renders_dir(env_key: str | None = None) -> Path:
    """Pixel dumps for viewer / smoke tests."""
    path = RENDERS_DIR if env_key is None else RENDERS_DIR / env_key
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_gl_backend(*, viewer: bool) -> None:
    """Set MUJOCO_GL before stable_worldmodel / dm_control import.

    MuJoCo viewer mode keeps EGL for offscreen policy pixels and only switches
    to GLFW when opening launch_passive.
    """
    import os

    if viewer:
        os.environ["MUJOCO_GL"] = "egl"
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")


def enable_glfw_for_viewer() -> None:
    """Switch to GLFW immediately before launch_passive."""
    import os

    os.environ["MUJOCO_GL"] = "glfw"
