"""High-level viewer session for World evaluation."""

from __future__ import annotations

from live_viewer.backends import resolve_backend, supports_viewer
from live_viewer.backends.base import ViewerBackend, ViewerHandle
from live_viewer.config import ViewerConfig


class WorldViewer:
    """Attach an interactive viewer to a stable_worldmodel World."""

    def __init__(
        self,
        world,
        *,
        backend: ViewerBackend,
        handle: ViewerHandle,
        config: ViewerConfig,
    ):
        self.world = world
        self.backend = backend
        self.handle = handle
        self.config = config

    @classmethod
    def prepare(
        cls,
        world,
        *,
        config: ViewerConfig | None = None,
        env_key: str = "env",
    ) -> tuple[ViewerBackend, ViewerHandle | None, ViewerConfig]:
        """Install backend hooks (e.g. EGL offscreen render) before env hooks that render."""
        cfg = config or ViewerConfig(render_subdir=env_key)
        backend = resolve_backend(world)
        partial = backend.prepare(world, config=cfg)
        return backend, partial, cfg

    @classmethod
    def open_prepared(
        cls,
        world,
        backend: ViewerBackend,
        partial: ViewerHandle | None,
        *,
        seed: int,
        config: ViewerConfig,
    ) -> WorldViewer:
        handle = backend.open(world, partial, seed=seed, config=config)
        return cls(world=world, backend=backend, handle=handle, config=config)

    @classmethod
    def attach(
        cls,
        world,
        *,
        seed: int,
        config: ViewerConfig | None = None,
        env_key: str = "env",
    ) -> WorldViewer:
        """Prepare backend, reset world, and open the viewer window."""
        backend, partial, cfg = cls.prepare(world, config=config, env_key=env_key)
        return cls.open_prepared(world, backend, partial, seed=seed, config=cfg)

    def sync(self) -> None:
        self.backend.sync(self.world, self.handle)

    def render_pixels(self) -> object:
        return self.backend.render_pixels(
            self.world,
            self.handle,
            width=self.config.img_size,
            height=self.config.img_size,
        )

    def hold_until_closed(self) -> None:
        self.backend.hold_until_closed(self.handle)

    def close(self) -> None:
        self.backend.close(self.handle)

    def __enter__(self) -> WorldViewer:
        return self

    def __exit__(self, *_) -> None:
        self.close()


__all__ = ["WorldViewer", "supports_viewer"]
