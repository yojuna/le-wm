"""Dump-driven visualization for lewm-phi (spec 15).

PCA on real latents only; figures motivate a named scalar. Linear projection only.
"""

from __future__ import annotations

import json
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
    load_dump as _load_dump,
)

# Re-export so CA1 / Fig-3 share one definition.
contact_events = _contact_events

PUB_SIZE = (7.0, 4.0)


class ImaginedFitError(ValueError):
    """Raised when PCA is asked to fit imagined latents."""


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
    # (Q, 1, D) - (1, B, D)
    d = np.linalg.norm(q[:, None, :] - b[None, :, :], axis=-1)
    idx = d.argmin(axis=1)
    dist = d[np.arange(len(idx)), idx]
    if squeeze:
        return idx[0], dist[0]
    return idx, dist


def load_dump(path: Path | str) -> dict[str, Any]:
    """Phase-B dump.npz (version 1 or 2)."""
    return _load_dump(Path(path))


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


def annotate(fig, motivates: str, *, captured_variance: float | None = None, **scalars):
    fig._viz_motivates = motivates  # type: ignore[attr-defined]
    fig._viz_scalars = scalars  # type: ignore[attr-defined]
    fig._viz_captured_variance = captured_variance  # type: ignore[attr-defined]
    return fig


def save_figure(fig, stem: Path) -> None:
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


def fig_oracle_overlay(
    ca0: dict,
    *,
    pair: int = 0,
    m_open: int = 25,
    m_closed: int = 5,
    summary: dict | None = None,
):
    """Oracle-imagine overlay in a real-fitted PCA frame.

    Motivates: ‖ẑ_end−z*‖ and toward-goal fraction vs m (CA0 curve).
    """
    plt = _plt()
    z_true = np.asarray(ca0["z_true"][pair], dtype=np.float64)
    z_star = np.asarray(ca0["z_star"][pair], dtype=np.float64)
    m_values = [int(x) for x in ca0["m_values"]]
    z_hat = np.asarray(ca0["z_hat"][pair])  # (n_m, L, D)
    d_end = np.asarray(ca0["d_end"][pair])
    d_start = float(ca0["d_start"][pair])
    toward = np.asarray(ca0["toward"][pair])

    def _idx(m: int) -> int:
        if m in m_values:
            return m_values.index(m)
        return int(np.argmin([abs(x - m) for x in m_values]))

    i_ol, i_cl = _idx(m_open), _idx(m_closed)
    proj = RealFittedProjector(n_components=2)
    proj.fit(z_true)
    r = proj.transform(z_true)
    ol = proj.transform(z_hat[i_ol])
    cl = proj.transform(z_hat[i_cl])
    star = proj.transform(z_star)
    start = r[0]
    var = proj.captured_variance

    fig, ax = plt.subplots(figsize=PUB_SIZE)
    ax.plot(r[:, 0], r[:, 1], "-o", ms=3, label="real z", color="C0")
    ax.plot(ol[:, 0], ol[:, 1], "-s", ms=3, label=f"open-loop m={m_values[i_ol]}", color="C3")
    ax.plot(cl[:, 0], cl[:, 1], "-^", ms=3, label=f"closed-loop m={m_values[i_cl]}", color="C2")
    ax.scatter([start[0]], [start[1]], c="k", s=40, zorder=5, label="start")
    ax.scatter([star[0]], [star[1]], c="gold", s=60, marker="*", zorder=5, label="z*")
    ax.annotate("", xy=r[min(3, len(r) - 1)], xytext=r[0], arrowprops=dict(arrowstyle="->", color="C0"))
    ax.set_xlabel("PC1 (real-fit)")
    ax.set_ylabel("PC2 (real-fit)")
    ax.set_title(
        f"Fig-1 pair {pair}: PCA captured {var:.1%}  |  "
        f"‖ẑ_end−z*‖ m={m_values[i_ol]}={d_end[i_ol]:.2f}  "
        f"m={m_values[i_cl]}={d_end[i_cl]:.2f}  d_start={d_start:.2f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fork = (summary or {}).get("fork")
    return annotate(
        fig,
        "‖ẑ_end−z*‖ and toward-goal fraction vs m (CA0 curve)",
        captured_variance=var,
        d_end_open=float(d_end[i_ol]),
        d_end_closed=float(d_end[i_cl]),
        d_start=d_start,
        toward_open=bool(toward[i_ol]),
        toward_closed=bool(toward[i_cl]),
        fork=fork,
        pair=pair,
    )


def fig_cem_landscape(capture: dict):
    """CEM candidate costs with oracle overlay.

    Motivates: cost(oracle actions) − cost(CEM-selected) (planner regret).
    """
    plt = _plt()
    actions = np.asarray(capture["actions"], dtype=np.float64)
    costs = np.asarray(capture["costs"], dtype=np.float64).reshape(-1)
    selected = int(np.asarray(capture["selected_idx"]).reshape(-1)[0])
    S = actions.shape[0]
    flat = actions.reshape(S, -1)
    sel = flat[selected]
    dist = np.linalg.norm(flat - sel[None, :], axis=1)
    oracle_cost = capture.get("oracle_cost")
    oracle_flat = capture.get("oracle_action")
    regret = None
    if oracle_cost is not None:
        oracle_cost = float(np.asarray(oracle_cost).reshape(-1)[0])
        regret = float(oracle_cost - costs[selected])

    fig, ax = plt.subplots(figsize=PUB_SIZE)
    sc = ax.scatter(dist, costs, c=costs, cmap="viridis", s=12, alpha=0.7)
    fig.colorbar(sc, ax=ax, label="cost")
    ax.scatter([0.0], [costs[selected]], c="red", s=50, marker="x", label="CEM selected", zorder=5)
    if oracle_flat is not None and oracle_cost is not None:
        of = np.asarray(oracle_flat, dtype=np.float64).reshape(-1)
        if of.size == sel.size:
            od = float(np.linalg.norm(of - sel))
            ax.scatter([od], [oracle_cost], c="gold", s=70, marker="*", label="oracle", zorder=6)
    ax.set_xlabel("‖a − a_selected‖ (flattened CEM tokens)")
    ax.set_ylabel("planning cost")
    title = f"Fig-2: CEM landscape  selected_cost={costs[selected]:.3g}"
    if regret is not None:
        title += f"  regret(oracle−sel)={regret:.3g}"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return annotate(
        fig,
        "cost(oracle actions) − cost(CEM-selected) (planner regret)",
        selected_cost=float(costs[selected]),
        oracle_cost=float(oracle_cost) if oracle_cost is not None else None,
        regret=regret,
        n_candidates=int(S),
    )


def fig_drift_contacts(dump: dict, *, segment: int = 0, env: str = "pusht"):
    """Per-step ‖ẑ−z‖ with contact/wall bands.

    Motivates: free-space vs contact mean-drift scalars.
    """
    plt = _plt()
    if "drift_true" in dump:
        drift_all = np.asarray(dump["drift_true"], dtype=np.float64)
        state = np.asarray(dump["state"])
        env = str((dump.get("meta") or {}).get("env") or env)
        d = drift_all[segment]
        st = state[segment]
        mean_d = drift_all.mean(axis=0)
    else:
        # CA0-style: z_true / z_hat[m=25]
        z_true = np.asarray(dump["z_true"])
        z_hat = np.asarray(dump["z_hat"])
        m_values = [int(x) for x in dump["m_values"]]
        mi = m_values.index(25) if 25 in m_values else int(np.argmax(m_values))
        drift_all = np.linalg.norm(z_hat[:, mi] - z_true, axis=-1)
        st = np.asarray(dump["path_state"][segment])
        d = drift_all[segment]
        mean_d = drift_all.mean(axis=0)

    ev = contact_events(st, env=env)
    ev_all = contact_events(
        dump["state"] if "state" in dump else dump["path_state"], env=env
    )
    pred = np.zeros_like(ev_all["any"], dtype=bool)
    if pred.ndim == 2 and pred.shape[1] > HISTORY:
        pred[:, HISTORY:] = True
    contact_mean = float(drift_all[ev_all["any"] & pred].mean()) if (ev_all["any"] & pred).any() else float("nan")
    free_mean = float(drift_all[(~ev_all["any"]) & pred].mean()) if ((~ev_all["any"]) & pred).any() else float("nan")

    h = np.arange(len(d))
    fig, ax = plt.subplots(figsize=PUB_SIZE)
    colors = np.where(ev["any"], "C3", "C0")
    ax.scatter(h, d, c=colors, s=18, zorder=3, label="this segment")
    ax.plot(h, mean_d, color="gray", lw=1, label="mean over segments")
    for t in np.where(ev["any"])[0]:
        ax.axvspan(t - 0.4, t + 0.4, color="C3", alpha=0.15)
    ax.axvline(HISTORY - 1, color="k", ls=":", lw=1, label="end of teacher-forced history")
    ax.set_xlabel("frame")
    ax.set_ylabel("‖ẑ − z‖₂")
    ax.set_title(
        f"Fig-3: drift vs contact  free={free_mean:.2f}  event={contact_mean:.2f}  "
        f"ratio={contact_mean / max(free_mean, 1e-8):.2f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return annotate(
        fig,
        "free-space vs contact mean-drift scalars",
        free_mean_drift=free_mean,
        contact_mean_drift=contact_mean,
        ratio=float(contact_mean / max(free_mean, 1e-8)),
        segment=segment,
    )


def fig_rollout_filmstrip(
    path_pixels: np.ndarray,
    z_true: np.ndarray,
    z_hat: np.ndarray,
    *,
    stride: int = 5,
    max_cols: int = 6,
):
    """Decoder-free: true pixels vs NN retrieval of ẑ in the true-z bank.

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
    cross = int(np.argmax(dist > thresh)) if np.any(dist > thresh) else int(L - 1)
    steps = list(range(0, L, stride))[:max_cols]
    n = len(steps)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.6))
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for j, t in enumerate(steps):
        true = _as_image(pix[t])
        retrieved = _as_image(pix[int(idx[t])])
        axes[0, j].imshow(true)
        axes[0, j].set_title(f"t={t} true", fontsize=8)
        axes[0, j].axis("off")
        axes[1, j].imshow(retrieved)
        axes[1, j].set_title(f"NN d={dist[t]:.2f}", fontsize=8)
        axes[1, j].axis("off")
    fig.suptitle(f"Fig-4 filmstrip  first d>median+std at t={cross}  (thresh={thresh:.2f})")
    fig.tight_layout()
    return annotate(
        fig,
        "step index where NN-retrieval distance crosses a threshold",
        nn_cross_step=cross,
        nn_threshold=thresh,
        mean_nn_dist=float(dist.mean()),
    )


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


def fig_probe_faithfulness(dump: dict, *, factor: str = "block_x"):
    """Ridge readout vs true factor, colored by board position.

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

    axes[0].set_title(
        f"R² all={_r2(r2_all)}  wall={_r2(r2_wall)}  center={_r2(r2_center)}"
    )
    axes[0].legend(fontsize=8)
    resid = pred - y
    sc = axes[1].scatter(block[:, 0], block[:, 1], c=np.abs(resid), s=6, cmap="magma")
    fig.colorbar(sc, ax=axes[1], label="|residual|")
    axes[1].set_xlabel("block_x")
    axes[1].set_ylabel("block_y")
    axes[1].set_title("Fig-5 residual on the board")
    fig.tight_layout()
    return annotate(
        fig,
        "conditional R² by region (near-wall vs center)",
        r2_all=r2_all,
        r2_wall=r2_wall,
        r2_center=r2_center,
        factor=factor,
    )


def fig_intervention_sweep(sweep: dict):
    """Predicted factor movement vs ε, free vs contact base states.

    Motivates: slope + linearity (R² of the ε→movement fit) per region.
    """
    plt = _plt()
    eps = np.asarray(sweep["eps"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=PUB_SIZE)
    scalars = {}
    for name, color in (("free", "C0"), ("contact", "C3")):
        y = np.asarray(sweep[name], dtype=np.float64)
        ax.plot(eps, y, "-o", ms=4, color=color, label=name)
        if len(eps) >= 2:
            coef = np.polyfit(eps, y, 1)
            yhat = np.polyval(coef, eps)
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            lin = 1.0 - ss_res / max(ss_tot, 1e-12)
            scalars[f"slope_{name}"] = float(coef[0])
            scalars[f"linearity_{name}"] = lin
            ax.plot(eps, yhat, "--", color=color, alpha=0.5)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("ε along probe direction")
    ax.set_ylabel("Δ predicted factor (one P step)")
    ax.set_title(
        "Fig-6 intervention sweep  "
        + "  ".join(f"{k}={v:.3f}" for k, v in scalars.items())
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return annotate(
        fig,
        "slope + linearity (R² of the ε→movement fit) per region",
        **scalars,
    )


def fig_rank_spectrum(dump: dict, *, factor: str = "block_x"):
    """Scree, per-dim variance, and pose R² vs #PCA components.

    Motivates: elbow index and dead-dim fraction (CA2).
    """
    plt = _plt()
    z = np.asarray(dump["z"], dtype=np.float64)
    flat = z.reshape(-1, z.shape[-1])
    centered = flat - flat.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eig = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))[::-1]
    pr = effective_rank(flat)
    total = eig.sum()
    cfrac = np.cumsum(eig) / max(total, 1e-12)
    elbow = int(np.searchsorted(cfrac, 0.9) + 1)
    dead = float((eig < 1e-3 * eig[0]).mean()) if eig[0] > 0 else 1.0
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
    axes[0].axvline(pr, color="C3", ls="--", label=f"participation ratio {pr:.1f}")
    axes[0].axvline(elbow, color="C2", ls=":", label=f"90% var k={elbow}")
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("eigenvalue")
    axes[0].set_title("scree")
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
    fig.suptitle(f"Fig-7 rank  dim={flat.shape[1]}  PR={pr:.1f}  elbow90={elbow}  dead={dead:.2f}")
    fig.tight_layout()
    return annotate(
        fig,
        "elbow index and dead-dim fraction (CA2)",
        participation_ratio=float(pr),
        elbow_90=elbow,
        dead_dim_fraction=dead,
        z_dim=int(flat.shape[1]),
    )


FIG_REGISTRY = {
    "oracle_overlay": fig_oracle_overlay,
    "cem_landscape": fig_cem_landscape,
    "drift_contacts": fig_drift_contacts,
    "rollout_filmstrip": fig_rollout_filmstrip,
    "probe_faithfulness": fig_probe_faithfulness,
    "intervention_sweep": fig_intervention_sweep,
    "rank_spectrum": fig_rank_spectrum,
}
