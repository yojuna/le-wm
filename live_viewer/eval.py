"""World.evaluate with an attached viewer."""

from __future__ import annotations

from pathlib import Path

from eval_logging import evaluate_logged
from eval_logging.logger import EvalRunLogger
from live_viewer.backends.base import ViewerClosed
from live_viewer.session import WorldViewer


def evaluate_with_viewer(
    world,
    viewer: WorldViewer,
    logger: EvalRunLogger,
    *,
    episodes: int,
    seed: int | None,
    render_dir: Path | None = None,
) -> dict:
    """Run evaluation with viewer sync and structured metrics."""
    return evaluate_logged(
        world,
        logger,
        episodes=episodes,
        seed=seed,
        render_dir=render_dir,
        render_fn=viewer.render_pixels if render_dir is not None else None,
        on_step_extra=lambda _w: viewer.sync(),
        interrupt_exceptions=(ViewerClosed,),
    )
