"""Viewer backend protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from live_viewer.config import ViewerConfig


class ViewerClosed(Exception):
    """Raised when the user closes the viewer window."""


@dataclass
class ViewerHandle:
    """Opaque per-backend state returned by open()."""

    backend: Any
    native: Any  # mujoco viewer handle, cv2 window name, etc.
    offscreen: Any | None = None
    extras: dict = field(default_factory=dict)


class ViewerBackend(ABC):
    """Interactive viewer for a stable_worldmodel World (env 0)."""

    name: str = "base"

    @classmethod
    @abstractmethod
    def supports_world(cls, world) -> bool:
        """Return True if this backend can attach to env 0."""

    @abstractmethod
    def prepare(self, world, *, config: ViewerConfig) -> ViewerHandle | None:
        """Install render patches etc. before the first reset. Returns partial handle."""

    @abstractmethod
    def open(
        self,
        world,
        handle: ViewerHandle | None,
        *,
        seed: int,
        config: ViewerConfig,
    ) -> ViewerHandle:
        """Reset world (if needed), open the window, return ready handle."""

    @abstractmethod
    def sync(self, world, handle: ViewerHandle) -> None:
        """Refresh the viewer after a simulation step."""

    @abstractmethod
    def render_pixels(self, world, handle: ViewerHandle, *, width: int, height: int):
        """Return an RGB frame for disk export (backend-specific)."""

    @abstractmethod
    def close(self, handle: ViewerHandle) -> None:
        """Release viewer and offscreen resources."""

    def hold_until_closed(self, handle: ViewerHandle) -> None:
        """Keep window open after eval (override per backend)."""
        import time

        print("evaluation finished — close the viewer window or press Ctrl+C to exit")
        try:
            while self.is_running(handle):
                self.sync_idle(handle)
                time.sleep(0.02)
        except (ViewerClosed, KeyboardInterrupt):
            pass

    def is_running(self, handle: ViewerHandle) -> bool:
        return bool(handle.native)

    def sync_idle(self, handle: ViewerHandle) -> None:
        """Sync without a world step (post-eval hold loop)."""
        if handle.native is not None and hasattr(handle.native, "sync"):
            if not handle.native.is_running():
                raise ViewerClosed()
            handle.native.sync()
