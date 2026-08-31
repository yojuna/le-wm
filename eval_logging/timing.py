"""Transparent wrapper that records CEM / solver latency."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


class TimedSolver:
    """Delegate to a solver while recording each solve() duration."""

    def __init__(self, solver: Any, on_solve: Callable[[float], None]) -> None:
        self._solver = solver
        self._on_solve = on_solve

    def configure(self, **kwargs: Any) -> None:
        return self._solver.configure(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def solve(self, info_dict: dict, init_action=None) -> dict:
        start = time.perf_counter()
        try:
            return self._solver.solve(info_dict, init_action=init_action)
        finally:
            self._on_solve(time.perf_counter() - start)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._solver, name)


def wrap_solver_timing(solver: Any, on_solve: Callable[[float], None]) -> TimedSolver:
    return TimedSolver(solver, on_solve)
