"""2D pygame window for pixel-renderable envs (PushT, etc.).

PushT uses Pymunk + pygame, not MuJoCo. We mirror the same stack for display
instead of OpenCV highgui (often headless in pip opencv builds).
"""

from __future__ import annotations

import numpy as np

from live_viewer.backends.base import ViewerBackend, ViewerClosed, ViewerHandle
from live_viewer.config import ViewerConfig


def _frame_from_world(world, *, width: int, height: int) -> np.ndarray:
    """Best-effort RGB frame (H, W, 3) uint8 from env 0."""
    if world.infos and "pixels" in world.infos:
        pixels = world.infos["pixels"]
        if isinstance(pixels, np.ndarray) and pixels.size:
            frame = np.asarray(pixels[0, 0])
            if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[0] != frame.shape[-1]:
                frame = np.transpose(frame, (1, 2, 0))
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            return frame

    env = world.envs.envs[0]
    frame = env.unwrapped.render()
    if frame is None:
        frame = env.render()
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def _rgb_to_pygame_surface(frame: np.ndarray):
    import pygame

    # pygame surfarray expects (width, height, 3)
    arr = np.transpose(frame, (1, 0, 2))
    return pygame.surfarray.make_surface(arr)


class PixelWindowBackend(ViewerBackend):
    """Show sim pixels in a pygame window."""

    name = "pixel_window"

    @classmethod
    def supports_world(cls, world) -> bool:
        env = world.envs.envs[0].unwrapped
        if not callable(getattr(env, "render", None)):
            return False
        modes = getattr(env, "metadata", {}).get("render_modes", [])
        return "rgb_array" in modes or "human" in modes

    def prepare(self, world, *, config: ViewerConfig) -> ViewerHandle | None:
        return None

    def open(
        self,
        world,
        handle: ViewerHandle | None,
        *,
        seed: int,
        config: ViewerConfig,
    ) -> ViewerHandle:
        import pygame

        world.reset(seed=seed)
        size = config.window_size or config.img_size
        pygame.init()
        pygame.display.init()
        screen = pygame.display.set_mode((size, size))
        pygame.display.set_caption("lewm viewer (pixel)")
        handle = ViewerHandle(
            backend=self,
            native=screen,
            extras={"open": True, "w": size, "h": size},
        )
        self.sync(world, handle)
        return handle

    def sync(self, world, handle: ViewerHandle) -> None:
        import pygame

        if not handle.extras.get("open", True):
            raise ViewerClosed()

        w = handle.extras.get("w", 224)
        h = handle.extras.get("h", 224)
        frame = _frame_from_world(world, width=w, height=h)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        surf = _rgb_to_pygame_surface(frame)
        screen = handle.native
        if surf.get_size() != screen.get_size():
            surf = pygame.transform.smoothscale(surf, screen.get_size())
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                handle.extras["open"] = False
                raise ViewerClosed()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                handle.extras["open"] = False
                raise ViewerClosed()

    def render_pixels(self, world, handle: ViewerHandle, *, width: int, height: int):
        return _frame_from_world(world, width=width, height=height)

    def sync_idle(self, handle: ViewerHandle) -> None:
        import pygame

        if not self.is_running(handle):
            raise ViewerClosed()
        pygame.time.wait(20)
        for event in pygame.event.get():
            if event.type in (pygame.QUIT, pygame.KEYDOWN) and (
                event.type == pygame.QUIT
                or event.key == pygame.K_q
            ):
                handle.extras["open"] = False
                raise ViewerClosed()

    def close(self, handle: ViewerHandle) -> None:
        import pygame

        handle.extras["open"] = False
        if handle.native is not None:
            try:
                pygame.display.quit()
            except Exception:
                pass

    def is_running(self, handle: ViewerHandle) -> bool:
        return bool(handle.extras.get("open", True))
