"""Viewer backend registry."""

from __future__ import annotations

from live_viewer.backends.base import ViewerBackend, ViewerClosed, ViewerHandle
from live_viewer.backends.mujoco_dmc import MujocoDmcBackend, get_dmc_physics
from live_viewer.backends.pixel_window import PixelWindowBackend

# Order matters: prefer 3D MuJoCo when available.
BACKENDS: tuple[type[ViewerBackend], ...] = (
    MujocoDmcBackend,
    PixelWindowBackend,
)


def resolve_backend(world) -> ViewerBackend:
    """Pick the first backend that supports env 0."""
    for cls in BACKENDS:
        if cls.supports_world(world):
            return cls()
    raise RuntimeError(
        "No viewer backend supports this environment. "
        "Need dm_control physics or an env with render() -> rgb_array."
    )


def supports_viewer(world) -> bool:
    try:
        resolve_backend(world)
        return True
    except RuntimeError:
        return False
