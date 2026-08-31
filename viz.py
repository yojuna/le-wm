"""Dump-driven visualization for lewm-phi (spec 15 v3).

PCA on real latents only. Every fig_* returns a FigureResult. Linear projection only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from phase_b import (
    HISTORY,
    contact_events as _contact_events,
    effective_rank,
    factor_names_for_env,
    load_dump as _load_dump,
)

contact_events = _contact_events

PUB_SIZE = (7.0, 4.0)
THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.yaml"


class ImaginedFitError(ValueError):
    """Raised when PCA is asked to fit imagined latents."""


@dataclass
class FigureResult:
    """Contract so report.py can assemble a scientific diagnostic without a gallery."""

    figure: Any
    tier: int
    question: str
    scalars: dict[str, dict[str, Any]]
    caption: dict[str, str]
    motivates: str = ""

    def attach(self) -> "FigureResult":
        fig = self.figure
        fig._viz_motivates = self.motivates or self.question
        fig._viz_scalars = {
            k: (v.get("value") if isinstance(v, dict) else v) for k, v in self.scalars.items()
        }
        cap = self.scalars.get("captured_variance") or self.scalars.get("pca_captured_variance")
        fig._viz_captured_variance = (
            cap.get("value") if isinstance(cap, dict) else cap
        )
        fig._viz_result = self
        return self


def load_thresholds(path: Path | str | None = None) -> dict:
    p = Path(path) if path else THRESHOLDS_PATH
    text = p.read_text()
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> dict:
    root: dict = {}
    section = root
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            key = line[:-1].strip()
            root[key] = {}
            section = root[key]
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            k, v = k.strip(), v.strip()
            if not v:
                continue
            try:
                section[k] = float(v) if "." in v else int(v)
            except ValueError:
                section[k] = v
    return root


def _sv(value, threshold=None, verdict=None) -> dict:
    return {"value": value, "threshold": threshold, "verdict": verdict}


class RealFittedProjector:
    """PCA fit on real z only; transform anything (including ẑ)."""

    def __init__(self, n_components: int = 2):
        self.n_components = int(n_components)
        self._pca: PCA | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self._cache_key: tuple | None = None

    def fit(self, z_real: np.ndarray, *, imagined: bool = False, cache_key=None):
        if imagined:
            raise ImaginedFitError(
                "RealFittedProjector refuses to fit on imagined latents "
                "(fit on real z, then transform ẑ)"
            )
        z = np.asarray(z_real, dtype=np.float64)
        if z.ndim == 3:
            z = z.reshape(-1, z.shape[-1])
        if z.ndim != 2:
            raise ValueError(f"expected 2-D or 3-D real z, got {z.shape}")
        self._pca = PCA(n_components=self.n_components)
        self._pca.fit(z)
        self.explained_variance_ratio_ = np.asarray(
            self._pca.explained_variance_ratio_, dtype=np.float64
        )
        self._cache_key = cache_key
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("call fit() on real latents first")
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            return self._pca.transform(arr.reshape(1, -1))[0]
        if arr.ndim == 3:
            flat = arr.reshape(-1, arr.shape[-1])
            out = self._pca.transform(flat)
            return out.reshape(*arr.shape[:-1], self.n_components)
        return self._pca.transform(arr)

    @property
    def captured_variance(self) -> float:
        if self.explained_variance_ratio_ is None:
            return float("nan")
        return float(self.explained_variance_ratio_.sum())


def nn_retrieve(
    z_query: np.ndarray, bank: np.ndarray, metric: str = "l2"
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest encoded frames. Returns (indices, distances)."""
    q = np.asarray(z_query, dtype=np.float64)
    b = np.asarray(bank, dtype=np.float64)
    if q.ndim == 1:
        q = q.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    if metric != "l2":
        raise ValueError(f"only metric='l2' is implemented, got {metric!r}")
    d = np.linalg.norm(q[:, None, :] - b[None, :, :], axis=-1)
    idx = d.argmin(axis=1)
    dist = d[np.arange(len(idx)), idx]
    if squeeze:
        return idx[0], dist[0]
    return idx, dist


def _capn(x, nd: int = 2) -> str:
    """Caption number: no raw None, no 12-digit floats."""
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        if x != x:
            return "—"
        return f"{x:.{nd}f}"
    return str(x)


def probe_decompose(vec: np.ndarray, dirs: dict[str, np.ndarray]) -> dict[str, float]:
    """Share of ||vec||² along named unit directions."""
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    denom = float(np.dot(v, v)) + 1e-12
    out = {}
    for name, d in dirs.items():
        u = np.asarray(d, dtype=np.float64).reshape(-1)
        n = np.linalg.norm(u)
        if n < 1e-12 or u.size != v.size:
            out[name] = 0.0
            continue
        u = u / n
        out[name] = float((np.dot(v, u) ** 2) / denom)
    return out


def probe_energy_shares(delta: np.ndarray, dirs: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean per-step share of ||Δz||² along named unit directions (no signed cancel)."""
    arr = np.asarray(delta, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    denom = np.square(arr).sum(axis=-1) + 1e-12
    out = {}
    for name, d in dirs.items():
        u = np.asarray(d, dtype=np.float64).reshape(-1)
        n = np.linalg.norm(u)
        if n < 1e-12 or u.size != arr.shape[-1]:
            out[name] = 0.0
            continue
        u = u / n
        out[name] = float(np.mean((arr @ u) ** 2 / denom))
    return out


def _mean_angle_deg(real_d: np.ndarray, imag_d: np.ndarray) -> float | None:
    angs = []
    for a, b in zip(np.asarray(real_d), np.asarray(imag_d)):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            continue
        c = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
        angs.append(float(np.degrees(np.arccos(c))))
    return float(np.mean(angs)) if angs else None


def _bank_label(dump: dict) -> str:
    meta = dump.get("meta") or {}
    if meta.get("collector"):
        return f"{meta.get('env', 'env')}/{meta.get('collector')}"
    if "z_true" in dump and "m_values" in dump:
        return "ca0-livebank"
    return str(meta.get("env") or "unknown")


def nonlinear_id(z: np.ndarray, *, max_points: int = 400, seed: int = 0) -> float:
    """TwoNN intrinsic-dimension estimate (Levina–Bickel / Facco). Linear scree can miss a curve."""
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    n = len(arr)
    if n < 8:
        return float("nan")
    rng = np.random.default_rng(seed)
    if n > max_points:
        arr = arr[rng.choice(n, size=max_points, replace=False)]
    d = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = np.sort(d, axis=1)[:, :2]
    r1 = np.clip(nn[:, 0], 1e-12, None)
    mu = nn[:, 1] / r1
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    if mu.size < 4:
        return float("nan")
    return float(1.0 / np.mean(np.log(mu)))


def action_effect(z_t: np.ndarray, z_next: np.ndarray) -> np.ndarray:
    """One-step latent displacement (real or imagined)."""
    return np.asarray(z_next, dtype=np.float64) - np.asarray(z_t, dtype=np.float64)


def load_dump(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        path = path / "dump.npz"
    return _load_dump(path)


def load_ca0(path: Path | str) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    npz = path / "ca0.npz" if path.is_dir() else path
    blob = np.load(npz, allow_pickle=True)
    data = {k: blob[k] for k in blob.files}
    summary_path = npz.parent / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return data, summary


def load_cem_capture(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    npz = path / "cem_capture.npz" if path.is_dir() else path
    blob = np.load(npz, allow_pickle=True)
    data = {k: blob[k] for k in blob.files}
    meta_path = npz.with_suffix(".meta.json")
    if not meta_path.exists():
        meta_path = npz.parent / "cem_capture.meta.json"
    data["meta"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return data


def as_figure(obj) -> Any:
    return obj.figure if isinstance(obj, FigureResult) else obj


def save_figure(fig, stem: Path) -> None:
    fig = as_figure(fig)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=120, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _m_index(m_values, m: int) -> int:
    m_values = [int(x) for x in m_values]
    if m in m_values:
        return m_values.index(m)
    return int(np.argmin([abs(x - m) for x in m_values]))


def ca0_fork_from_by_m(by_m: dict, thresholds: dict | None = None) -> dict:
    """Same cuts as scripts/closed_loop_imagine.decide_fork (do not retune)."""
    th = (thresholds or load_thresholds()).get("ca0", {})
    m1_tg, m1_d = th.get("m1_toward_guard", 0.90), th.get("m1_d_end_guard", 1.0)
    a_tg, a_d = th.get("accum_toward_at_m5", 0.60), th.get("accum_d_end_at_m5", 3.0)
    i_tg, i_d = th.get("infidelity_toward_at_m3", 0.60), th.get("infidelity_d_end_at_m3", 5.0)

    def _row(m):
        row = by_m.get(m) or by_m.get(str(m))
        if row is None:
            return None
        if "frac_toward" not in row and "mean_d_end" in row:
            return row
        return row

    m1, m3, m5 = _row(1), _row(3), _row(5)
    if m1 is not None:
        toward = float(m1.get("frac_toward", m1.get("toward", 0)))
        dend = float(m1["mean_d_end"])
        if toward < m1_tg or dend > m1_d:
            return {
                "fork": "CA0-INFIDELITY",
                "reason": (
                    f"m=1 not near-perfect (toward={toward:.3f}, d_end={dend:.3f}) "
                    "— single-step prediction fails the guard."
                ),
            }
    if m5 is not None:
        toward = float(m5.get("frac_toward", 0))
        dend = float(m5["mean_d_end"])
        if toward >= a_tg and dend <= a_d:
            return {
                "fork": "CA0-ACCUMULATION",
                "reason": f"m=5 toward={toward:.3f} (>= {a_tg}) and d_end={dend:.3f} (<= {a_d})",
            }
    if m3 is not None:
        toward = float(m3.get("frac_toward", 0))
        dend = float(m3["mean_d_end"])
        if toward < i_tg or dend > i_d:
            return {
                "fork": "CA0-INFIDELITY",
                "reason": f"drift persists at m=3 (toward={toward:.3f}, d_end={dend:.3f})",
            }
    return {
        "fork": "CA0-AMBIGUOUS",
        "reason": "m=5 did not meet ACCUMULATION cuts; m=3 did not meet INFIDELITY cuts.",
    }


def fig_a1(
    ca0: dict,
    *,
    pair: int = 0,
    m_open: int = 25,
    m_closed: int = 5,
    summary: dict | None = None,
) -> FigureResult:
    """Oracle-imagine overlay + full-space distance curve.

    Motivates: ‖ẑ_end−z*‖ and toward-goal fraction vs m (CA0 curve).
    """
    plt = _plt()
    th = load_thresholds()
    m_values = [int(x) for x in ca0["m_values"]]
    i_ol, i_cl = _m_index(m_values, m_open), _m_index(m_values, m_closed)
    z_true = np.asarray(ca0["z_true"][pair], dtype=np.float64)
    z_star = np.asarray(ca0["z_star"][pair], dtype=np.float64)
    z_hat = np.asarray(ca0["z_hat"][pair])
    d_end_pair = np.asarray(ca0["d_end"][pair])
    d_start_pair = float(ca0["d_start"][pair])
    toward_pair = np.asarray(ca0["toward"][pair])
    d_end_all = np.asarray(ca0["d_end"], dtype=np.float64)
    d_start_all = np.asarray(ca0["d_start"], dtype=np.float64)
    toward_all = np.asarray(ca0["toward"], dtype=np.float64)
    mean_d_end_open = float(d_end_all[:, i_ol].mean())
    mean_d_end_closed = float(d_end_all[:, i_cl].mean())
    mean_d_start = float(d_start_all.mean())
    frac_toward_open = float(toward_all[:, i_ol].mean())
    frac_toward_closed = float(toward_all[:, i_cl].mean())
    frac_pairs_drift = float((d_end_all[:, i_ol] > d_start_all).mean())
    proj = RealFittedProjector(n_components=2)
    proj.fit(z_true)
    r = proj.transform(z_true)
    ol = proj.transform(z_hat[i_ol])
    cl = proj.transform(z_hat[i_cl])
    star = proj.transform(z_star)
    var = proj.captured_variance

    d_full = np.linalg.norm(z_hat[i_ol] - z_true, axis=-1)
    d_plane = np.linalg.norm(ol - r, axis=-1)
    pred = slice(HISTORY, None) if len(d_full) > HISTORY else slice(None)
    in_plane = float(np.mean(d_plane[pred] / np.clip(d_full[pred], 1e-8, None)))
    d_star_ol = np.linalg.norm(z_hat[i_ol] - z_star[None, :], axis=-1)
    d_star_cl = np.linalg.norm(z_hat[i_cl] - z_star[None, :], axis=-1)
    d_star_true = np.linalg.norm(z_true - z_star[None, :], axis=-1)
    deco_cut = float(th.get("a1", {}).get("in_plane_decorative_below", 0.35))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.plot(r[:, 0], r[:, 1], "-o", ms=3, label="real z", color="C0")
    ax.plot(ol[:, 0], ol[:, 1], "-s", ms=3, label=f"open-loop m={m_values[i_ol]}", color="C3")
    ax.plot(cl[:, 0], cl[:, 1], "-^", ms=3, label=f"closed-loop m={m_values[i_cl]}", color="C2")
    ax.scatter([r[0, 0]], [r[0, 1]], c="k", s=40, zorder=5, label="start")
    ax.scatter([star[0]], [star[1]], c="gold", s=60, marker="*", zorder=5, label="z*")
    ax.annotate("", xy=r[min(3, len(r) - 1)], xytext=r[0], arrowprops=dict(arrowstyle="->", color="C0"))
    ax.set_xlabel("PC1 (real-fit)")
    ax.set_ylabel("PC2 (real-fit)")
    ax.set_title(f"example pair {pair}  PCA {var:.0%}  in-plane {in_plane:.2f}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    t = np.arange(len(d_star_ol))
    ax2.plot(t, d_star_true, color="C0", label="‖z_t−z*‖")
    ax2.plot(t, d_star_ol, color="C3", label=f"‖ẑ_t−z*‖ m={m_values[i_ol]}")
    ax2.plot(t, d_star_cl, color="C2", label=f"‖ẑ_t−z*‖ m={m_values[i_cl]}")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("full-space distance")
    ax2.set_title(f"example pair {pair}  full-space ‖·−z*‖")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fork = (summary or {}).get("fork")
    decorative = in_plane < deco_cut
    return FigureResult(
        figure=fig,
        tier=1,
        question="Does the rollout drift, and how?",
        motivates="‖ẑ_end−z*‖ and toward-goal fraction vs m (CA0 curve)",
        scalars={
            "d_end_open": _sv(mean_d_end_open),
            "d_end_closed": _sv(mean_d_end_closed),
            "d_start": _sv(mean_d_start),
            "frac_toward_open": _sv(frac_toward_open),
            "frac_toward_closed": _sv(frac_toward_closed),
            "frac_pairs_drift": _sv(frac_pairs_drift),
            "example_d_end_open": _sv(float(d_end_pair[i_ol])),
            "example_d_start": _sv(d_start_pair),
            "example_toward_open": _sv(bool(toward_pair[i_ol])),
            "pca_captured_variance": _sv(var),
            "in_plane_drift_fraction": _sv(in_plane, deco_cut, "decorative" if decorative else "informative"),
            "fork": _sv(fork),
            "example_pair": _sv(pair),
            "n_pairs": _sv(int(len(d_end_all))),
            "bank": _sv("ca0-livebank"),
        },
        caption={
            "what": "Bank-mean ‖ẑ_end−z*‖ is the citable scalar. Panels are one labeled example pair in a PCA fit on that pair's real z, paired with that pair's full-space distance to z*.",
            "how_to_read": "Overshoot of z* suggests calibration; wander orthogonal from step 1 suggests action mis-representation; track-then-diverge suggests accumulation. If in-plane fraction is small, trust the right panel. Do not read the example end-dist as the bank mean.",
            "reading_here": (
                f"Bank mean open-loop end-dist {_capn(mean_d_end_open)} vs start {_capn(mean_d_start)} "
                f"({frac_pairs_drift:.0%} of pairs drift). "
                f"Example pair {pair}: {_capn(float(d_end_pair[i_ol]))} vs {_capn(d_start_pair)}; "
                f"in-plane {_capn(in_plane)} ({'decorative overlay' if decorative else 'overlay informative'})."
            ),
            "would_overturn": "Bank-mean open-loop ẑ staying at or below d_start would overturn the drift claim.",
        },
    ).attach()


fig_oracle_overlay = fig_a1


def fig_a2(ca0: dict, *, summary: dict | None = None) -> FigureResult:
    """Single-step error histogram + CA0 recovery vs m.

    Motivates: median single-step error; accumulation slope; recovery-m.
    """
    plt = _plt()
    th = load_thresholds()
    z_true = np.asarray(ca0["z_true"], dtype=np.float64)
    z_hat = np.asarray(ca0["z_hat"])
    m_values = [int(x) for x in ca0["m_values"]]
    i1 = _m_index(m_values, 1)
    hat1 = z_hat[:, i1]
    err = np.linalg.norm(hat1[:, HISTORY:] - z_true[:, HISTORY:], axis=-1).reshape(-1)
    median = float(np.median(err))
    adj = np.linalg.norm(z_true[:, 1:] - z_true[:, :-1], axis=-1).reshape(-1)
    median_adj = float(np.median(adj)) if adj.size else None
    floor = (summary or {}).get("encoder_floor") or {}
    same_state = floor.get("same_state_reencode_median")
    d_end = np.asarray(ca0["d_end"], dtype=np.float64)
    toward = np.asarray(ca0["toward"], dtype=np.float64)
    mean_dend = d_end.mean(axis=0)
    mean_toward = toward.mean(axis=0)
    slope = float(np.polyfit(m_values, mean_dend, 1)[0]) if len(m_values) >= 2 else float("nan")
    ca0th = th.get("ca0", {})
    recovery_m = None
    for j, m in enumerate(m_values):
        if mean_toward[j] >= ca0th.get("accum_toward_at_m5", 0.6) and mean_dend[j] <= ca0th.get(
            "accum_d_end_at_m5", 3.0
        ):
            recovery_m = int(m)
            break
    by_m = {}
    if summary and summary.get("by_m"):
        raw = summary["by_m"]
        by_m = {int(k): v for k, v in raw.items()}
    else:
        for j, m in enumerate(m_values):
            by_m[int(m)] = {
                "frac_toward": float(mean_toward[j]),
                "mean_d_end": float(mean_dend[j]),
            }
    fork = ca0_fork_from_by_m(by_m, th)
    onestep_cut = float(th.get("a2", {}).get("median_onestep_abs", 0.8))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].hist(err, bins=40, color="C0", edgecolor="none")
    axes[0].axvline(median, color="C3", ls="--", label=f"median {median:.2f}")
    axes[0].set_xlabel("teacher-forced ‖ẑ_{t+1}−z_{t+1}‖ (m=1)")
    axes[0].set_ylabel("count")
    axes[0].set_title("single-step error")
    axes[0].legend(fontsize=8)
    ax = axes[1]
    ax.plot(m_values, mean_dend, "-o", color="C3", label="mean ‖ẑ_end−z*‖")
    ax.set_xlabel("re-encode interval m")
    ax.set_ylabel("mean end-dist", color="C3")
    ax2 = ax.twinx()
    ax2.plot(m_values, mean_toward, "-s", color="C2", label="toward-goal frac")
    ax2.set_ylabel("toward-goal", color="C2")
    ax.set_title(f"recovery vs m  slope={slope:.2f}  {fork['fork']}")
    fig.tight_layout()
    return FigureResult(
        figure=fig,
        tier=1,
        question="Accumulation or per-step infidelity?",
        motivates="median single-step error; accumulation slope; recovery-m",
        scalars={
            "median_onestep_error": _sv(median, onestep_cut, "fail" if median > onestep_cut else "pass"),
            "median_adjacent_true_z": _sv(median_adj),
            "onestep_over_adjacent": _sv(
                None if not median_adj else float(median / max(median_adj, 1e-8))
            ),
            "same_state_reencode_median": _sv(same_state),
            "accumulation_slope": _sv(slope),
            "recovery_m": _sv(recovery_m),
            "fork": _sv(fork["fork"]),
            "fork_reason": _sv(fork["reason"]),
            "bank": _sv("ca0-livebank"),
        },
        caption={
            "what": "Teacher-forced one-step errors (m=1) beside the CA0 recovery curve over re-encode interval m. Adjacent true-z is the scale of one real step (motion + two encodes), not same-frame encoder noise.",
            "how_to_read": "Large single-step error → per-step infidelity (retrain). Tiny single-step and a steep drop as m shrinks → accumulation (protocol). Compare one-step error to adjacent true-z; same-state re-encode is the instrument floor under the m=1 guard.",
            "reading_here": (
                f"Median one-step {_capn(median)} vs adjacent true-z {_capn(median_adj)}; "
                f"same-state re-encode {_capn(same_state)}. "
                f"recovery-m={recovery_m}; fork {fork['fork']}."
            ),
            "would_overturn": "m=1 toward-goal ≥ 0.90 and end-dist ≤ 1.0 would pass the teacher-force guard and reopen the accumulation reading. A same-state re-encode floor near that 1.0 cut would mean the guard is below the instrument.",
        },
    ).attach()


def fig_b1(capture: dict) -> FigureResult:
    """Terrain-free CEM panels: cost vs distance, line-search, candidate PCA.

    Motivates: signed cost gap = cost(oracle) − cost(CEM-selected) (planner regret).
    """
    plt = _plt()
    th = load_thresholds()
    actions = np.asarray(capture["actions"], dtype=np.float64)
    costs = np.asarray(capture["costs"], dtype=np.float64).reshape(-1)
    selected = int(np.asarray(capture["selected_idx"]).reshape(-1)[0])
    S = actions.shape[0]
    flat = actions.reshape(S, -1)
    sel = flat[selected]
    dist = np.linalg.norm(flat - sel[None, :], axis=1)
    oc_raw = capture.get("oracle_cost")
    oracle_cost = None
    if oc_raw is not None:
        v = float(np.asarray(oc_raw).reshape(-1)[0])
        if np.isfinite(v):
            oracle_cost = v
    regret = None if oracle_cost is None else float(oracle_cost - costs[selected])
    gap_cut = float(th.get("b1", {}).get("signed_gap_model_if_gt", 0.0))
    gap_verdict = None
    if regret is not None:
        gap_verdict = "model" if regret > gap_cut else "search"

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    ax = axes[0]
    sc = ax.scatter(dist, costs, c=costs, cmap="viridis", s=10, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="cost")
    ax.scatter([0.0], [costs[selected]], c="red", s=50, marker="x", label="selected", zorder=5)
    oracle_flat = capture.get("oracle_action")
    if oracle_flat is not None and oracle_cost is not None:
        of = np.asarray(oracle_flat, dtype=np.float64).reshape(-1)
        if of.size == sel.size:
            ax.scatter(
                [float(np.linalg.norm(of - sel))],
                [oracle_cost],
                c="gold",
                s=70,
                marker="*",
                label="oracle",
                zorder=6,
            )
    ax.set_xlabel("‖a − a_selected‖")
    ax.set_ylabel("cost")
    ax.set_title("B1a cost vs distance")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    alphas = capture.get("line_search_alphas")
    ls = capture.get("line_search_costs")
    if alphas is not None and ls is not None and np.asarray(ls).size > 1:
        a = np.asarray(alphas, dtype=np.float64).reshape(-1)
        c = np.asarray(ls, dtype=np.float64).reshape(-1)
        ax.plot(a, c, "-o", ms=4, color="C0")
        ax.set_title("B1b selected→oracle")
        slope_ls = float(c[-1] - c[0])
    else:
        ax.text(0.5, 0.5, "no line-search in capture", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("B1b selected→oracle")
        slope_ls = None
    ax.set_xlabel("α (0=selected, 1=oracle)")
    ax.set_ylabel("cost")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    pca = PCA(n_components=2)
    xy = pca.fit_transform(flat)
    var = float(pca.explained_variance_ratio_.sum())
    ax.scatter(xy[:, 0], xy[:, 1], c=costs, cmap="viridis", s=10, alpha=0.7)
    ax.scatter([xy[selected, 0]], [xy[selected, 1]], c="red", s=50, marker="x", zorder=5, label="selected")
    if oracle_flat is not None and np.asarray(oracle_flat).reshape(-1).size == sel.size:
        oxy = pca.transform(np.asarray(oracle_flat, dtype=np.float64).reshape(1, -1))[0]
        ax.scatter([oxy[0]], [oxy[1]], c="gold", s=70, marker="*", zorder=6, label="oracle")
    ax.set_xlabel("PC1 (candidates)")
    ax.set_ylabel("PC2")
    ax.set_title(f"B1c action PCA  var={var:.0%}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return FigureResult(
        figure=fig,
        tier=1,
        question="Search problem or model problem?",
        motivates="cost(oracle actions) − cost(CEM-selected) (planner regret)",
        scalars={
            "selected_cost": _sv(float(costs[selected])),
            "oracle_cost": _sv(oracle_cost),
            "signed_cost_gap": _sv(regret, gap_cut, gap_verdict),
            "n_candidates": _sv(int(S)),
            "pca_captured_variance": _sv(var),
            "line_search_delta": _sv(slope_ls),
        },
        caption={
            "what": "Final CEM candidate set: cost vs distance, interpolation toward the oracle, PCA of actions.",
            "how_to_read": "Bowl vs flat plate vs rugged in B1a. Line-search down toward the oracle = search failed to find; up = the model scores the good action worse.",
            "reading_here": (
                f"signed gap {regret:.2f} ({gap_verdict})"
                if regret is not None
                else "oracle cost missing"
            ),
            "would_overturn": "A negative signed gap with line-search decreasing toward the oracle would read as search, not model.",
        },
    ).attach()


fig_cem_landscape = fig_b1


def fig_b2(capture: dict) -> FigureResult:
    """CEM elite cost vs iteration.

    Motivates: iters-to-plateau; final elite gap to oracle.
    """
    plt = _plt()
    best = np.asarray(capture.get("iter_best", []), dtype=np.float64).reshape(-1)
    mean = np.asarray(capture.get("iter_mean", []), dtype=np.float64).reshape(-1)
    std = np.asarray(capture.get("iter_std", []), dtype=np.float64).reshape(-1)
    oc = capture.get("oracle_cost")
    oracle_cost = float(np.asarray(oc).reshape(-1)[0]) if oc is not None else None
    fig, ax = plt.subplots(figsize=PUB_SIZE)
    plateau = None
    elite_gap = None
    if best.size:
        ax.plot(best, "-o", ms=3, label="best", color="C3")
        if mean.size:
            ax.plot(mean, "-s", ms=3, label="mean", color="C0")
        if std.size == mean.size and mean.size:
            ax.fill_between(np.arange(len(mean)), mean - std, mean + std, color="C0", alpha=0.15)
        diffs = np.abs(np.diff(best))
        if diffs.size:
            rel = diffs / max(float(np.abs(best[0])), 1e-6)
            hit = np.where(rel < 0.01)[0]
            plateau = int(hit[0] + 1) if len(hit) else int(len(best))
        if oracle_cost is not None:
            elite_gap = float(best[-1] - oracle_cost)
            ax.axhline(oracle_cost, color="gold", ls="--", label="oracle")
        ax.set_xlabel("CEM iteration")
        ax.set_ylabel("cost")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"B2 convergence  plateau~{plateau}  elite−oracle={elite_gap}")
    else:
        ax.text(0.5, 0.5, "no per-iteration stats in capture", ha="center")
        ax.set_title("B2 convergence")
    fig.tight_layout()
    still = bool(best.size >= 2 and abs(best[-1] - best[-2]) > 0.01 * max(abs(best[-1]), 1e-6))
    if oracle_cost is not None and best.size and oracle_cost > float(best[-1]):
        still_verdict = "wrong-objective" if still else "converged-wrong-objective"
    elif still:
        still_verdict = "under-budget"
    else:
        still_verdict = "converged"
    return FigureResult(
        figure=fig,
        tier=2,
        question="Did CEM give up, or converge confidently to a bad optimum?",
        motivates="iters-to-plateau; final elite gap to oracle",
        scalars={
            "iters_to_plateau": _sv(plateau),
            "final_elite_gap_to_oracle": _sv(elite_gap),
            "still_improving_at_end": _sv(still, None, still_verdict),
        },
        caption={
            "what": "Elite best/mean CEM cost across iterations (not the full candidate tensor).",
            "how_to_read": "If the oracle costs *more* than the CEM pick, further iterations descend a mis-ranked objective and move *away* from the good action — that is not an under-budget search problem. Under-budget only applies when the oracle is cheaper than the elite and the curve is still falling.",
            "reading_here": (
                f"plateau ~{plateau}; still improving={still} ({still_verdict}); "
                f"elite−oracle={_capn(elite_gap)}."
            ),
            "would_overturn": "A late drop that *reaches* oracle cost would read as under-budget search, not a model-scoring fault.",
        },
    ).attach()


def fig_b3(capture: dict) -> FigureResult:
    """Per-horizon-step selected vs oracle.

    Motivates: per-step |selected−oracle|.
    """
    plt = _plt()
    actions = np.asarray(capture["actions"])
    selected = int(np.asarray(capture["selected_idx"]).reshape(-1)[0])
    sel = actions[selected]
    ora = np.asarray(capture.get("oracle_action", np.zeros(0)))
    fig, ax = plt.subplots(figsize=PUB_SIZE)
    per_step = None
    if ora.size and ora.reshape(-1).size == sel.reshape(-1).size:
        ora = ora.reshape(sel.shape)
        per_step = np.linalg.norm(sel - ora, axis=-1)
        ax.plot(per_step, "-o", ms=4)
        ax.set_xlabel("horizon step")
        ax.set_ylabel("‖selected−oracle‖")
        ax.set_title("B3 per-step deviation")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "oracle action shape mismatch", ha="center")
    fig.tight_layout()
    first_bad = None
    if per_step is not None and per_step.size:
        med = float(np.median(per_step))
        hits = np.where(per_step > med * 1.5)[0]
        first_bad = int(hits[0]) if len(hits) else None
    mean_dev = None if per_step is None else float(per_step.mean())
    if first_bad is None and per_step is not None:
        reading = (
            f"no single first-bad step; mean {_capn(mean_dev)} — whole plan off-oracle "
            "(not a late-lookahead miss)."
        )
    else:
        reading = f"first large deviation at step {first_bad}."
    return FigureResult(
        figure=fig,
        tier=3,
        question="Which action in the chunk is wrong?",
        motivates="per-step |selected−oracle|",
        scalars={
            "mean_step_deviation": _sv(mean_dev),
            "first_bad_step": _sv(first_bad),
        },
        caption={
            "what": "Horizon-step deviation between CEM-selected tokens and packed oracle tokens.",
            "how_to_read": "First-right/later-wrong → receding-horizon is fine, lookahead is the problem. Uniformly large deviation → the whole chunk is off-oracle.",
            "reading_here": reading,
            "would_overturn": "Uniformly small per-step deviation would mean CEM found a near-oracle plan in action space.",
        },
    ).attach()


def _as_phase_b_like(dump: dict) -> dict:
    """Map CA0 keys onto phase-b names so A3 can decompose."""
    if "z" in dump and "state" in dump:
        return dump
    if "z_true" not in dump:
        return dump
    out = dict(dump)
    out["z"] = dump["z_true"]
    m_values = [int(x) for x in dump["m_values"]]
    mi = _m_index(m_values, 25)
    out["z_hat"] = np.asarray(dump["z_hat"])[:, mi]
    if "path_state" in dump:
        out["state"] = dump["path_state"]
    if "factor_names" not in out and "state" in out:
        st = np.asarray(out["state"])
        out["factor_names"] = np.array(factor_names_for_env("pusht", st.shape[-1]))
    return out


def _drift_and_state(dump: dict):
    dump = _as_phase_b_like(dump)
    env = str((dump.get("meta") or {}).get("env") or "pusht")
    if "drift_true" in dump:
        return np.asarray(dump["drift_true"], dtype=np.float64), np.asarray(dump["state"]), env
    if "z_true" in dump:
        z_true = np.asarray(dump["z_true"])
        z_hat = np.asarray(dump["z_hat"])
        if z_hat.ndim == 4:
            m_values = [int(x) for x in dump["m_values"]]
            mi = _m_index(m_values, 25)
            z_hat = z_hat[:, mi]
        drift_all = np.linalg.norm(z_hat - z_true, axis=-1)
        return drift_all, np.asarray(dump.get("path_state", dump.get("state"))), env
    z = np.asarray(dump["z"])
    hat = np.asarray(dump["z_hat"])
    drift_all = np.linalg.norm(hat - z, axis=-1)
    return drift_all, np.asarray(dump["state"]), env


def _factor_dirs(dump: dict) -> dict[str, np.ndarray]:
    dump = _as_phase_b_like(dump)
    if "z" not in dump or "state" not in dump:
        return {}
    z = np.asarray(dump["z"], dtype=np.float64)
    state = np.asarray(dump["state"], dtype=np.float64)
    names = [str(x) for x in np.asarray(dump["factor_names"]).tolist()]
    zf = z.reshape(-1, z.shape[-1])
    st = state.reshape(-1, state.shape[-1])
    dirs = {}
    want = ["block_x", "block_y", "block_angle", "agent_x", "agent_y"]
    for name in want:
        if name not in names:
            continue
        y = st[:, names.index(name)]
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        pipe.fit(zf, y)
        w = pipe.named_steps["ridge"].coef_ / np.clip(pipe.named_steps["standardscaler"].scale_, 1e-8, None)
        n = np.linalg.norm(w)
        dirs[name] = w / n if n > 1e-12 else w
    return dirs


def fig_a3(dump: dict, *, segment: int = 0, env: str = "pusht") -> FigureResult:
    """Per-step ‖ẑ−z‖ with contact bands and directional decompose.

    Motivates: free-space vs contact mean-drift scalars; per-factor drift share.
    """
    plt = _plt()
    th = load_thresholds()
    drift_all, state, env = _drift_and_state(dump)
    env = str((dump.get("meta") or {}).get("env") or env)
    d = drift_all[segment]
    st = state[segment]
    mean_d = drift_all.mean(axis=0)
    ev = contact_events(st, env=env)
    ev_all = contact_events(state, env=env)
    pred = np.zeros_like(ev_all["any"], dtype=bool)
    if pred.ndim == 2 and pred.shape[1] > HISTORY:
        pred[:, HISTORY:] = True
    cmask = ev_all["any"] & pred
    fmask = (~ev_all["any"]) & pred
    contact_mean = float(drift_all[cmask].mean()) if cmask.any() else None
    free_mean = float(drift_all[fmask].mean()) if fmask.any() else None
    ratio = None if not free_mean or not contact_mean else float(contact_mean / max(free_mean, 1e-8))
    spike_cut = float(th.get("a3", {}).get("contact_ratio_spike_if_gt", 1.5))

    dump_n = _as_phase_b_like(dump)
    dirs = _factor_dirs(dump_n)
    shares = {}
    if "z" in dump_n and "z_hat" in dump_n and dirs:
        z = np.asarray(dump_n["z"])
        hat = np.asarray(dump_n["z_hat"])
        if hat.shape == z.shape:
            delta = hat - z
            if delta.ndim == 3 and delta.shape[1] > HISTORY:
                delta = delta[:, HISTORY:]
            shares = probe_energy_shares(delta.reshape(-1, z.shape[-1]), dirs)

    bank = _bank_label(dump_n)
    h = np.arange(len(d))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    colors = np.where(ev["any"], "C3", "C0")
    ax.scatter(h, d, c=colors, s=18, zorder=3, label="this segment")
    ax.plot(h, mean_d, color="gray", lw=1, label="mean")
    for t in np.where(ev["any"])[0]:
        ax.axvspan(t - 0.4, t + 0.4, color="C3", alpha=0.15)
    ax.axvline(HISTORY - 1, color="k", ls=":", lw=1)
    ax.set_xlabel("frame")
    ax.set_ylabel("‖ẑ − z‖₂")
    ax.set_title(f"example seg {segment}  free={_capn(free_mean)}  contact={_capn(contact_mean)}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    if shares:
        ax.bar(list(shares.keys()), list(shares.values()), color="C0")
        ax.set_ylabel("mean per-step share of ‖Δz‖²")
        ax.set_title("probe energy shares")
        ax.tick_params(axis="x", rotation=30)
    else:
        ax.text(0.5, 0.5, "no factor dirs (need dump z)", ha="center", transform=ax.transAxes)
        ax.set_title("probe_decompose")
    fig.tight_layout()
    block_share = float(shares.get("block_x", 0) + shares.get("block_y", 0)) if shares else None
    agent_share = float(shares.get("agent_x", 0) + shares.get("agent_y", 0)) if shares else None
    pose_share = None
    if shares:
        pose_share = float(sum(shares.values()))
    return FigureResult(
        figure=fig,
        tier=2,
        question="Why does it drift — contact vs steady, in which factor?",
        motivates="free-space vs contact mean-drift scalars; per-factor drift energy share",
        scalars={
            "free_mean_drift": _sv(free_mean),
            "contact_mean_drift": _sv(contact_mean),
            "ratio_contact_over_free": _sv(
                ratio, spike_cut, "spike" if ratio is not None and ratio > spike_cut else "smooth"
            ),
            "block_xy_share": _sv(block_share),
            "agent_xy_share": _sv(agent_share),
            "pose_probe_share_sum": _sv(pose_share),
            "segment": _sv(segment),
            "bank": _sv(bank),
        },
        caption={
            "what": f"Per-step prediction error with contact/wall bands, plus mean per-step energy along linear pose-probe directions. Bank: {bank}. Example segment {segment} in the left panel; contact/free scalars are over the whole dump.",
            "how_to_read": "Jumps at red bands → contact-representation fault. Pose-probe shares near zero → drift lives outside the linear pose subspace (not 'lost the block' vs 'lost the agent').",
            "reading_here": (
                f"contact/free ratio {_capn(ratio)}; block_xy {_capn(block_share, 3)}; "
                f"agent_xy {_capn(agent_share, 3)}; pose-probe sum {_capn(pose_share, 3)}."
            ),
            "would_overturn": "A ratio ≫ 1.5, or pose-probe shares dominating ‖Δz‖², would overturn a 'generic / off-pose-subspace' reading.",
        },
    ).attach()


fig_drift_contacts = fig_a3


def fig_a4(ca0: dict, actions: np.ndarray) -> FigureResult:
    """One-step error vs |action|.

    Motivates: error–|action| correlation; in-box vs at-bound error.
    """
    plt = _plt()
    z_true = np.asarray(ca0["z_true"], dtype=np.float64)
    z_hat = np.asarray(ca0["z_hat"])
    m_values = [int(x) for x in ca0["m_values"]]
    hat = z_hat[:, _m_index(m_values, 1)]
    err = np.linalg.norm(hat[:, HISTORY:] - z_true[:, HISTORY:], axis=-1)
    acts = np.asarray(actions, dtype=np.float64)
    if acts.ndim == 2:
        acts = acts[None, ...]
    # align: actions (N, L, A) or (N, L-?, A)
    n = min(len(err), len(acts))
    err_n = err[:n]
    acts_n = acts[:n]
    l_err, l_act = err_n.shape[1], acts_n.shape[1]
    if l_act >= HISTORY + l_err:
        mag = np.linalg.norm(acts_n[:, HISTORY : HISTORY + l_err], axis=-1)
    elif l_act >= l_err:
        mag = np.linalg.norm(acts_n[:, :l_err], axis=-1)
    else:
        mag = np.linalg.norm(acts_n, axis=-1)
        err_n = err_n[:, : mag.shape[1]]
    e = err_n.reshape(-1)
    m = mag.reshape(-1)
    nuse = min(e.size, m.size)
    e, m = e[:nuse], m[:nuse]
    corr = float(np.corrcoef(e, m)[0, 1]) if nuse > 4 and m.std() > 1e-8 else None
    bound = np.percentile(m, 90) if nuse else 0
    in_box = float(e[m < bound].mean()) if nuse and (m < bound).any() else None
    at_b = float(e[m >= bound].mean()) if nuse and (m >= bound).any() else None
    fig, ax = plt.subplots(figsize=PUB_SIZE)
    ax.scatter(m, e, s=8, alpha=0.4)
    ax.set_xlabel("|action|")
    ax.set_ylabel("one-step ‖ẑ−z‖")
    ax.set_title(f"A4 error vs |a|  corr={_capn(corr)}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return FigureResult(
        figure=fig,
        tier=3,
        question="Is fidelity uniform, or regime-limited by action size?",
        motivates="error–|action| correlation; in-box vs at-bound error",
        scalars={"error_action_corr": _sv(corr), "error_in_box": _sv(in_box), "error_at_bound": _sv(at_b)},
        caption={
            "what": "Teacher-forced one-step error against action magnitude on oracle paths.",
            "how_to_read": "A rising cloud at large |a| means the model is worse at the action boundary.",
            "reading_here": f"corr={_capn(corr)}; in-box {_capn(in_box)} vs bound {_capn(at_b)}.",
            "would_overturn": "Near-zero correlation with equal in-box and bound error would say fidelity is not action-regime limited.",
        },
    ).attach()


def fig_a5(ca0: dict, *, pair: int = 0) -> FigureResult:
    """Real vs imagined one-step displacement.

    Motivates: mean full-space angular error (bank); PCA quiver is the example view.
    """
    plt = _plt()
    z_all = np.asarray(ca0["z_true"], dtype=np.float64)
    hat_all = np.asarray(ca0["z_hat"][:, _m_index(ca0["m_values"], 1)], dtype=np.float64)
    pair_angles = []
    step_angles = []
    for i in range(len(z_all)):
        real_d = np.diff(z_all[i], axis=0)
        imag_d = hat_all[i, 1:] - z_all[i, :-1]
        ang_i = _mean_angle_deg(real_d, imag_d)
        if ang_i is not None:
            pair_angles.append(ang_i)
        for a, b in zip(real_d, imag_d):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                continue
            c = float(np.clip(np.dot(a, b) / (na * nb), -1, 1))
            step_angles.append(float(np.degrees(np.arccos(c))))
    bank_full = float(np.mean(pair_angles)) if pair_angles else None
    bank_full_steps = float(np.mean(step_angles)) if step_angles else None

    z_true = z_all[pair]
    hat = hat_all[pair]
    proj = RealFittedProjector(n_components=2)
    proj.fit(z_true)
    r = proj.transform(z_true)
    h = proj.transform(hat)
    real_d = np.diff(r, axis=0)
    imag_d = h[1:] - r[:-1]
    plane_ex = _mean_angle_deg(real_d, imag_d)
    full_ex = _mean_angle_deg(np.diff(z_true, axis=0), hat[1:] - z_true[:-1])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    ax.quiver(r[:-1, 0], r[:-1, 1], real_d[:, 0], real_d[:, 1], color="C0", angles="xy", scale_units="xy", scale=1, width=0.004, label="real")
    ax.quiver(r[:-1, 0], r[:-1, 1], imag_d[:, 0], imag_d[:, 1], color="C3", angles="xy", scale_units="xy", scale=1, width=0.004, label="imagined")
    ax.set_title(f"example pair {pair}  PCA {proj.captured_variance:.0%}  plane {_capn(plane_ex)}°")
    ax.legend(fontsize=8)
    ax.set_xlabel("PC1 (real-fit)")
    ax.set_ylabel("PC2")
    ax = axes[1]
    if pair_angles:
        ax.hist(pair_angles, bins=20, color="C0", edgecolor="none")
        ax.axvline(bank_full, color="C3", ls="--", label=f"mean {_capn(bank_full)}°")
        ax.legend(fontsize=8)
    ax.set_xlabel("full-space angle per pair (deg)")
    ax.set_ylabel("count")
    ax.set_title("honest scalar (full z, all pairs)")
    fig.tight_layout()
    return FigureResult(
        figure=fig,
        tier=2,
        question="Does the model get action direction right?",
        motivates="mean full-space angular error over the bank",
        scalars={
            "mean_angular_error_deg": _sv(bank_full),
            "mean_angular_error_deg_steps": _sv(bank_full_steps),
            "example_plane_angle_deg": _sv(plane_ex),
            "example_full_angle_deg": _sv(full_ex),
            "pca_captured_variance": _sv(proj.captured_variance),
            "example_pair": _sv(pair),
            "n_pairs": _sv(int(len(pair_angles))),
            "bank": _sv("ca0-livebank"),
        },
        caption={
            "what": "Left: example-pair quiver in a real-fitted PCA plane (lossy). Right: per-pair mean angle in full z — the citable scalar.",
            "how_to_read": "Aligned arrows → direction is right even if speed is wrong. Trust the histogram, not the 2D shadow. A ~40° full-space error is a systematically wrong action→next-state map, not isotropic jitter.",
            "reading_here": (
                f"Bank-mean full-space angle {_capn(bank_full)}° "
                f"(steps {_capn(bank_full_steps)}°). Example pair {pair}: plane {_capn(plane_ex)}° vs full {_capn(full_ex)}°."
            ),
            "would_overturn": "Bank-mean full-space angle near 0° would say the model gets action direction right.",
        },
    ).attach()


def fig_f1(
    path_pixels: np.ndarray,
    z_true: np.ndarray,
    z_hat: np.ndarray,
    *,
    stride: int = 5,
    max_cols: int = 6,
) -> FigureResult:
    """Decoder-free qualitative filmstrip: true pixels vs NN retrieval of ẑ.

    Motivates: the step index where NN-retrieval distance crosses a threshold.
    """
    plt = _plt()
    pix = np.asarray(path_pixels)
    z_true = np.asarray(z_true, dtype=np.float64)
    z_hat = np.asarray(z_hat, dtype=np.float64)
    L = min(len(pix), len(z_true), len(z_hat))
    idx, dist = nn_retrieve(z_hat[:L], z_true[:L])
    idx = np.atleast_1d(idx)
    dist = np.atleast_1d(dist)
    thresh = float(np.median(dist) + dist.std())
    large = dist > thresh
    cross = int(np.argmax(large)) if np.any(large) else int(L - 1)
    steps = list(range(0, L, stride))[:max_cols]
    n = len(steps)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.8))
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for j, t in enumerate(steps):
        axes[0, j].imshow(_as_image(pix[t]))
        axes[0, j].set_title(f"t={t} true", fontsize=8)
        axes[0, j].axis("off")
        flag = " !" if large[t] else ""
        axes[1, j].imshow(_as_image(pix[int(idx[t])]))
        axes[1, j].set_title(f"NN d={dist[t]:.2f}{flag}", fontsize=8)
        axes[1, j].axis("off")
    fig.suptitle(f"F1 qualitative filmstrip  first large NN at t={cross}")
    fig.tight_layout()
    return FigureResult(
        figure=fig,
        tier=3,
        question="Where physically does imagination diverge? (qualitative)",
        motivates="step index where NN-retrieval distance crosses a threshold",
        scalars={
            "nn_cross_step": _sv(cross, thresh),
            "nn_threshold": _sv(thresh),
            "mean_nn_dist": _sv(float(dist.mean())),
            "frac_large_retrieval": _sv(float(large.mean())),
        },
        caption={
            "what": "True frames beside the nearest real encoded frame to ẑ_t. Qualitative / exploratory — not a decoder.",
            "how_to_read": "A large NN distance means the model is imagining a state unlike any real frame in this episode.",
            "reading_here": f"first large retrieval at t={cross} (thresh={_capn(thresh)}).",
            "would_overturn": "NN distances staying small through the horizon would mean imagined latents remain on the real episode manifold.",
        },
    ).attach()


fig_rollout_filmstrip = fig_f1


def _as_image(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        a = arr.astype(np.float64)
        if a.max() <= 1.5:
            a = a * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    return arr


def fig_c2(dump: dict, *, factor: str = "block_x") -> FigureResult:
    """Ridge readout vs true factor, residual on the board.

    Motivates: conditional R² by region (near-wall vs center).
    """
    plt = _plt()
    z = np.asarray(dump["z"])
    state = np.asarray(dump["state"])
    names = [str(x) for x in np.asarray(dump["factor_names"]).tolist()]
    if factor not in names:
        factor = "block_x" if "block_x" in names else names[min(2, len(names) - 1)]
    fi = names.index(factor)
    n, l, d = z.shape
    zf = z.reshape(n * l, d)
    y = state.reshape(n * l, state.shape[-1])[:, fi]
    block = state.reshape(n * l, state.shape[-1])[:, 2:4]
    ev = contact_events(state.reshape(n * l, state.shape[-1]), env="pusht")
    wall = ev["wall"].reshape(-1)
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe.fit(zf, y)
    pred = pipe.predict(zf)
    r2_all = float(r2_score(y, pred))
    r2_wall = float(r2_score(y[wall], pred[wall])) if wall.any() else None
    r2_center = float(r2_score(y[~wall], pred[~wall])) if (~wall).any() else None
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    axes[0].scatter(y[~wall], pred[~wall], s=6, alpha=0.4, label="center", c="C0")
    if wall.any():
        axes[0].scatter(y[wall], pred[wall], s=8, alpha=0.5, label="near-wall", c="C3")
    lo, hi = float(y.min()), float(y.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel(f"true {factor}")
    axes[0].set_ylabel(f"probe {factor}")

    def _r2(x):
        return "n/a" if x is None else f"{x:.2f}"

    axes[0].set_title(f"R² all={_r2(r2_all)}  wall={_r2(r2_wall)}  center={_r2(r2_center)}")
    axes[0].legend(fontsize=8)
    resid = pred - y
    sc = axes[1].scatter(block[:, 0], block[:, 1], c=np.abs(resid), s=6, cmap="magma")
    fig.colorbar(sc, ax=axes[1], label="|residual|")
    axes[1].set_xlabel("block_x")
    axes[1].set_ylabel("block_y")
    axes[1].set_title("C2 residual on the board")
    fig.tight_layout()
    return FigureResult(
        figure=fig,
        tier=3,
        question="Where does block-pose legibility fail physically?",
        motivates="conditional R² by region (near-wall vs center)",
        scalars={
            "r2_all": _sv(r2_all),
            "r2_wall": _sv(r2_wall),
            "r2_center": _sv(r2_center),
            "factor": _sv(factor),
            "bank": _sv(_bank_label(dump)),
        },
        caption={
            "what": "Linear probe of a pose factor vs truth, with residual heat on the board.",
            "how_to_read": "Wall vs center R² split shows whether legibility is spatially uniform.",
            "reading_here": (
                f"R² all={_capn(r2_all)} wall="
                f"{'no wall-band samples in this bank' if r2_wall is None else _capn(r2_wall)} "
                f"center={_capn(r2_center)}. In-sample fit on {_bank_label(dump)} — not the episode-holdout probe."
            ),
            "would_overturn": "A large wall/center R² gap would mean legibility (and likely planning) fails in a physical region.",
        },
    ).attach()


fig_probe_faithfulness = fig_c2


def fig_d2(sweep: dict) -> FigureResult:
    """One-step ε sweep (full D2 multi-step/combine is writeup-only).

    Motivates: slope + linearity (R² of the ε→movement fit) per region.
    """
    plt = _plt()
    eps = np.asarray(sweep["eps"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=PUB_SIZE)
    raw = {}
    for name, color in (("free", "C0"), ("contact", "C3")):
        y = np.asarray(sweep[name], dtype=np.float64)
        ax.plot(eps, y, "-o", ms=4, color=color, label=name)
        if len(eps) >= 2:
            coef = np.polyfit(eps, y, 1)
            yhat = np.polyval(coef, eps)
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            lin = 1.0 - ss_res / max(ss_tot, 1e-12)
            raw[f"slope_{name}"] = float(coef[0])
            raw[f"linearity_{name}"] = lin
            ax.plot(eps, yhat, "--", color=color, alpha=0.5)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("ε along probe direction")
    ax.set_ylabel("Δ predicted factor (one P step)")
    ax.set_title("D2 one-step sweep (not multi-step / combine)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return FigureResult(
        figure=fig,
        tier=3,
        question="Is legible geometry cleanly actionable in one P step?",
        motivates="slope + linearity (R² of the ε→movement fit) per region",
        scalars={k: _sv(v) for k, v in raw.items()},
        caption={
            "what": "Predicted factor movement vs intervention size, free vs contact. One P step only — not the full D2 persistence/combine suite.",
            "how_to_read": "Linear monotone → usable for latent steering. Kink or contact-only saturation → bounded usable region.",
            "reading_here": " ".join(f"{k}={v:.3f}" for k, v in raw.items()),
            "would_overturn": "A contact-only kink or near-zero slope would bound latent-space steering, especially near contact.",
        },
    ).attach()


fig_intervention_sweep = fig_d2


def fig_d1(dump: dict, *, factor: str = "block_x") -> FigureResult:
    """Scree + dead-dim histogram + nonlinear ID.

    Motivates: elbow index and dead-dim fraction (CA2); nonlinear ID.
    """
    plt = _plt()
    th = load_thresholds()
    z = np.asarray(dump["z"], dtype=np.float64)
    flat = z.reshape(-1, z.shape[-1])
    centered = flat - flat.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eig = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))[::-1]
    pr = effective_rank(flat)
    nid = nonlinear_id(flat)
    total = eig.sum()
    cfrac = np.cumsum(eig) / max(total, 1e-12)
    elbow = int(np.searchsorted(cfrac, 0.9) + 1)
    dead = float((eig < 1e-3 * eig[0]).mean()) if eig[0] > 0 else 1.0
    dead_cut = float(th.get("d1", {}).get("dead_dim_hard_elbow_if_gt", 0.4))
    var = flat.var(axis=0)
    names = [str(x) for x in np.asarray(dump["factor_names"]).tolist()]
    state = np.asarray(dump["state"])
    y = None
    if factor in names:
        y = state.reshape(-1, state.shape[-1])[:, names.index(factor)]
    pca = PCA()
    pca.fit(flat)
    ks = list(range(1, min(40, flat.shape[1]) + 1))
    r2s = []
    if y is not None:
        z_pca = pca.transform(flat)
        for k in ks:
            pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            pipe.fit(z_pca[:, :k], y)
            r2s.append(float(r2_score(y, pipe.predict(z_pca[:, :k]))))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6))
    axes[0].semilogy(np.arange(1, len(eig) + 1), eig + 1e-18, marker=".", ms=3)
    axes[0].axvline(pr, color="C3", ls="--", label=f"PR {pr:.1f}")
    axes[0].axvline(elbow, color="C2", ls=":", label=f"90% k={elbow}")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("eigenvalue")
    axes[0].set_title(f"scree  nID={nid:.1f} (linear scree can miss a curve)")
    axes[0].legend(fontsize=7)
    axes[1].hist(var, bins=30, color="C0")
    axes[1].set_xlabel("per-dim variance")
    axes[1].set_title(f"dead-dim frac {dead:.2f}")
    if r2s:
        axes[2].plot(ks, r2s, "-o", ms=3)
        axes[2].axvline(pr, color="C3", ls="--")
        axes[2].set_xlabel("# PCA components")
        axes[2].set_ylabel(f"Ridge R² {factor}")
        axes[2].set_title("decodability vs rank")
    else:
        axes[2].text(0.5, 0.5, "no factor", ha="center")
    fig.suptitle(f"D1 rank  dim={flat.shape[1]}  PR={pr:.1f}  nID={nid:.1f}  dead={dead:.2f}")
    fig.tight_layout()
    return FigureResult(
        figure=fig,
        tier=2,
        question="Is capacity/structure the issue (should we scale)?",
        motivates="elbow index and dead-dim fraction (CA2)",
        scalars={
            "participation_ratio": _sv(float(pr)),
            "nonlinear_id": _sv(float(nid)),
            "elbow_90": _sv(elbow),
            "dead_dim_fraction": _sv(dead, dead_cut, "hard-elbow" if dead > dead_cut else "soft"),
            "z_dim": _sv(int(flat.shape[1])),
        },
        caption={
            "what": "Linear spectrum of z plus a TwoNN intrinsic-dimension estimate. Linear scree cannot see a curved manifold.",
            "how_to_read": "Sharp elbow, dead tail, and low nonlinear ID → genuine low-rank; do not scale the token.",
            "reading_here": f"PR={pr:.1f}, nonlinear ID={nid:.1f}, dead={dead:.2f}.",
            "would_overturn": "Nonlinear ID near the ambient 192 with a soft tail would reopen 'entanglement / unused capacity'.",
        },
    ).attach()


fig_rank_spectrum = fig_d1


def fig_e1(
    *,
    goal_xy: np.ndarray,
    success: np.ndarray,
    oracle_success: np.ndarray | None = None,
) -> FigureResult:
    """CEM success vs goal pose.

    Motivates: success by region; overlap fraction.
    """
    plt = _plt()
    xy = np.asarray(goal_xy, dtype=np.float64)
    suc = np.asarray(success, dtype=bool).reshape(-1)
    fig, ax = plt.subplots(figsize=PUB_SIZE)
    ax.scatter(xy[~suc, 0], xy[~suc, 1], c="C3", s=18, label="CEM fail", alpha=0.7)
    ax.scatter(xy[suc, 0], xy[suc, 1], c="C2", s=18, label="CEM ok", alpha=0.7)
    ax.set_xlabel("goal block_x")
    ax.set_ylabel("goal block_y")
    rate = float(suc.mean()) if suc.size else None
    ax.set_title(f"E1 success by goal pose  rate={rate}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return FigureResult(
        figure=fig,
        tier=3,
        question="Do failures cluster by goal pose?",
        motivates="success by region; overlap fraction",
        scalars={"cem_success_rate": _sv(rate), "n": _sv(int(suc.size))},
        caption={
            "what": "Target block poses colored by CEM success on the live-bank pairs.",
            "how_to_read": "Spatial clusters of failure that overlap high-residual (C2) or high-drift (A3) regions are one hard-region story.",
            "reading_here": f"CEM success {rate}.",
            "would_overturn": "Spatially uniform success would say failures are not a pose-region phenomenon.",
        },
    ).attach()


FIG_REGISTRY = {
    "a1": fig_a1,
    "oracle_overlay": fig_a1,
    "a2": fig_a2,
    "b1": fig_b1,
    "cem_landscape": fig_b1,
    "b2": fig_b2,
    "b3": fig_b3,
    "a3": fig_a3,
    "drift_contacts": fig_a3,
    "a4": fig_a4,
    "a5": fig_a5,
    "c2": fig_c2,
    "probe_faithfulness": fig_c2,
    "d1": fig_d1,
    "rank_spectrum": fig_d1,
    "d2": fig_d2,
    "intervention_sweep": fig_d2,
    "f1": fig_f1,
    "rollout_filmstrip": fig_f1,
    "e1": fig_e1,
}
