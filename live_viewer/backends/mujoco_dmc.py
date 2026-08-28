"""MuJoCo passive viewer for dm_control-based stable_worldmodel envs."""

from __future__ import annotations

from live_viewer.backends.base import ViewerBackend, ViewerClosed, ViewerHandle
from live_viewer.config import ViewerConfig, enable_glfw_for_viewer


def get_dmc_physics(world):
    """dm_control physics for env 0."""
    unwrapped = world.envs.envs[0].unwrapped
    dmc_env = getattr(unwrapped, "env", None)
    physics = getattr(dmc_env, "physics", None)
    if physics is None:
        raise RuntimeError("Not a dm_control env")
    return unwrapped, physics


class MujocoOffscreenRenderer:
    """Offscreen pixels via mujoco.Renderer (safe alongside launch_passive).

    dm_control's physics.render() steals the GLFW context; see mujoco #798.
    """

    def __init__(self, dmc_wrapper):
        import mujoco

        self._dmc = dmc_wrapper
        self._model = dmc_wrapper.env.physics.model.ptr
        self._data = dmc_wrapper.env.physics.data.ptr
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def render(self, width=224, height=224, camera_id=None):
        import mujoco

        key = (height, width)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(
                self._model, height=height, width=width
            )
        renderer = self._renderers[key]
        cid = camera_id if camera_id is not None else self._dmc.camera_id
        mujoco.mj_forward(self._model, self._data)
        renderer.update_scene(self._data, camera=cid)
        return renderer.render().copy()

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()

    def rebind(self, dmc_wrapper) -> None:
        """Reattach after MJCF recompile (first reset replaces model/data pointers)."""
        self.close()
        self._dmc = dmc_wrapper
        self._model = dmc_wrapper.env.physics.model.ptr
        self._data = dmc_wrapper.env.physics.data.ptr


class MujocoDmcBackend(ViewerBackend):
    name = "mujoco_dmc"

    @classmethod
    def supports_world(cls, world) -> bool:
        try:
            get_dmc_physics(world)
            return True
        except RuntimeError:
            return False

    def prepare(self, world, *, config: ViewerConfig) -> ViewerHandle:
        dmc, _ = get_dmc_physics(world)
        offscreen = MujocoOffscreenRenderer(dmc)
        dmc.render = offscreen.render
        return ViewerHandle(backend=self, native=None, offscreen=offscreen)

    def open(
        self,
        world,
        handle: ViewerHandle | None,
        *,
        seed: int,
        config: ViewerConfig,
    ) -> ViewerHandle:
        import mujoco.viewer

        offscreen = handle.offscreen if handle else None
        if offscreen is None:
            partial = self.prepare(world, config=ViewerConfig())
            offscreen = partial.offscreen

        world.reset(seed=seed)
        dmc, physics = get_dmc_physics(world)
        offscreen.rebind(dmc)

        enable_glfw_for_viewer()
        viewer = mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr)
        viewer.sync()
        return ViewerHandle(backend=self, native=viewer, offscreen=offscreen)

    def sync(self, world, handle: ViewerHandle) -> None:
        viewer = handle.native
        if not viewer.is_running():
            raise ViewerClosed()
        viewer.sync()

    def render_pixels(self, world, handle: ViewerHandle, *, width: int, height: int):
        if handle.offscreen is None:
            raise RuntimeError("offscreen renderer not installed")
        return handle.offscreen.render(width=width, height=height)

    def close(self, handle: ViewerHandle) -> None:
        if handle.native is not None:
            try:
                if handle.native.is_running():
                    handle.native.close()
            except Exception:
                pass
        if handle.offscreen is not None:
            handle.offscreen.close()

    def is_running(self, handle: ViewerHandle) -> bool:
        return handle.native is not None and handle.native.is_running()

    def sync_idle(self, handle: ViewerHandle) -> None:
        viewer = handle.native
        if viewer is None or not viewer.is_running():
            raise ViewerClosed()
        with viewer.lock():
            viewer.sync()
