#!/usr/bin/env python3
"""Where the ~40° smear lives, and whether it is systematic.

Dump-only. Cuts frozen in thresholds.yaml ``smear_structure`` *before* this
run. Does not start Part B.

  python scripts/smear_structure.py \\
      --ca0 eval_results/pusht/ca0_closed_loop/seed0 \\
      --oracle-bank eval_results/pusht/c0_oracle_livebank/seed0 \\
      --out eval_results/pusht/smear_structure/seed0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

from phase_b import HISTORY  # noqa: E402
from viz import load_ca0, load_thresholds  # noqa: E402


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if x != x else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def pca_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigvals descending, eigenvectors as columns, unit)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    c = arr - arr.mean(axis=0, keepdims=True)
    cov = np.cov(c, rowvar=False)
    eig, vec = np.linalg.eigh(cov)
    order = np.argsort(eig)[::-1]
    eig = np.maximum(eig[order], 0.0)
    return eig, vec[:, order]


def live_dead_masks(eig: np.ndarray, *, dead_rel: float, live_var_frac: float) -> dict:
    total = float(eig.sum()) + 1e-12
    dead = eig < float(dead_rel) * float(eig[0] if eig[0] > 0 else 1.0)
    cfrac = np.cumsum(eig) / total
    k90 = int(np.searchsorted(cfrac, float(live_var_frac)) + 1)
    live90 = np.zeros_like(eig, dtype=bool)
    live90[: max(k90, 1)] = True
    return {
        "k90": k90,
        "n_dead": int(dead.sum()),
        "dead_frac_dims": float(dead.mean()),
        "dead": dead,
        "live90": live90,
    }


def energy_share(vec: np.ndarray, basis: np.ndarray, mask: np.ndarray) -> float:
    """Mean share of ‖v‖² in the span of basis columns where mask is true."""
    cols = basis[:, mask]
    if cols.size == 0:
        return 0.0
    proj = vec @ cols
    num = np.square(proj).sum(axis=-1)
    den = np.square(vec).sum(axis=-1) + 1e-12
    return float(np.median(num / den))


def one_step_vectors(ca0: dict) -> dict:
    z = np.asarray(ca0["z_true"], dtype=np.float64)
    m_values = [int(x) for x in ca0["m_values"]]
    i1 = m_values.index(1) if 1 in m_values else 0
    hat = np.asarray(ca0["z_hat"][:, i1], dtype=np.float64)
    true_d = z[:, HISTORY:] - z[:, HISTORY - 1 : -1]
    pred_d = hat[:, HISTORY:] - z[:, HISTORY - 1 : -1]
    n, t, d = true_d.shape
    true_d = true_d.reshape(-1, d)
    pred_d = pred_d.reshape(-1, d)
    nt = np.linalg.norm(true_d, axis=-1)
    npd = np.linalg.norm(pred_d, axis=-1)
    mask = (nt > 1e-8) & (npd > 1e-8)
    u = np.zeros_like(true_d)
    u[mask] = true_d[mask] / nt[mask, None]
    par_len = np.einsum("ij,ij->i", pred_d, u)
    r = pred_d - u * par_len[:, None]
    pair_id = np.repeat(np.arange(n), t)
    t_id = np.tile(np.arange(t), n)
    return {
        "z_t": z[:, HISTORY - 1 : -1].reshape(-1, d),
        "true_d": true_d,
        "pred_d": pred_d,
        "r": r,
        "u": u,
        "nt": nt,
        "npd": npd,
        "par_len": par_len,
        "mask": mask,
        "pair_id": pair_id,
        "t_id": t_id,
        "n_pairs": n,
        "n_t": t,
        "path_state": np.asarray(ca0["path_state"], dtype=np.float64) if "path_state" in ca0 else None,
    }


def ridge_r2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    train: np.ndarray,
    basis: np.ndarray | None = None,
) -> float:
    """Held-out multivariate R² (1 − SSE/SST on test rows).

    If ``basis`` is given (columns), map x into that subspace first. Raw 192-d
    ridge on ~500 rows overfits; the positive control (true Δz → Δpose) then
    goes negative — that is an instrument failure, not a physics result.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if basis is not None:
        x = x @ np.asarray(basis, dtype=np.float64)
    tr, te = train, ~train
    if tr.sum() < max(x.shape[1] + 2, 8) or te.sum() < 8:
        return float("nan")
    mu = x[tr].mean(0)
    sd = x[tr].std(0) + 1e-8
    xs = (x - mu) / sd
    a = xs[tr].T @ xs[tr] + alpha * np.eye(xs.shape[1])
    w = np.linalg.solve(a, xs[tr].T @ y[tr])
    pred = xs[te] @ w
    yt = y[te]
    sst = np.square(yt - yt.mean(0, keepdims=True)).sum()
    sse = np.square(yt - pred).sum()
    if sst < 1e-12:
        return float("nan")
    return float(1.0 - sse / sst)


def knn_neighbor_cosine(
    feat: np.ndarray,
    vec: np.ndarray,
    pair_id: np.ndarray,
    t_id: np.ndarray,
    mask: np.ndarray,
    *,
    k: int,
    exclude_dt: int,
) -> dict:
    """Mean cosine of residual r among k nearest (state, action) neighbors."""
    idx = np.where(mask)[0]
    f = feat[idx]
    v = vec[idx]
    pid, tid = pair_id[idx], t_id[idx]
    mu, sd = f.mean(0), f.std(0) + 1e-8
    fs = (f - mu) / sd
    nrm = np.linalg.norm(v, axis=-1, keepdims=True)
    vn = v / np.clip(nrm, 1e-8, None)
    dmat = np.linalg.norm(fs[:, None, :] - fs[None, :, :], axis=-1)
    same = pid[:, None] == pid[None, :]
    close = same & (np.abs(tid[:, None] - tid[None, :]) <= int(exclude_dt))
    np.fill_diagonal(close, True)
    dmat = np.where(close, np.inf, dmat)
    kk = min(int(k), max(len(idx) - 1, 1))
    nn = np.argpartition(dmat, kk, axis=1)[:, :kk]
    cos = np.einsum("ij,ij->i", vn, vn[nn].mean(axis=1))
    # pairwise among the k neighbors: mean cosine of each point to its neighbors
    neigh_cos = []
    for i in range(len(idx)):
        nvec = vn[nn[i]]
        c = nvec @ vn[i]
        neigh_cos.append(float(np.mean(c)))
    arr = np.asarray(neigh_cos, dtype=np.float64)
    rng = np.random.default_rng(0)
    shuf = vn[rng.permutation(len(vn))]
    null = np.einsum("ij,ij->i", vn, shuf)
    return {
        "k": kk,
        "n": int(len(idx)),
        "mean_cosine": float(np.mean(arr)),
        "median_cosine": float(np.median(arr)),
        "null_shuffle_mean_cosine": float(np.mean(null)),
    }


def pose_delta(path_state: np.ndarray) -> np.ndarray:
    st = np.asarray(path_state, dtype=np.float64)
    d = st[:, HISTORY:] - st[:, HISTORY - 1 : -1]
    return d.reshape(-1, d.shape[-1])[:, :4]  # agent_xy + block_xy


def analyze_bank(ca0: dict, actions: np.ndarray | None, th: dict) -> dict:
    dead_rel = float(th["occupancy_dead_rel"])
    live_frac = float(th["occupancy_live_var_frac"])
    motion_frac = float(th["motion_var_frac"])
    k = int(th["knn_k"])
    excl = int(th["knn_exclude_dt"])

    step = one_step_vectors(ca0)
    m = step["mask"]
    r = step["r"][m]
    true_d = step["true_d"][m]
    pred_d = step["pred_d"][m]
    z_all = np.asarray(ca0["z_true"], dtype=np.float64).reshape(-1, np.asarray(ca0["z_true"]).shape[-1])

    eig_z, vz = pca_basis(z_all)
    occ = live_dead_masks(eig_z, dead_rel=dead_rel, live_var_frac=live_frac)
    eig_d, vd = pca_basis(true_d)
    mot = live_dead_masks(eig_d, dead_rel=dead_rel, live_var_frac=motion_frac)

    e_dead = energy_share(r, vz, occ["dead"])
    e_live = energy_share(r, vz, occ["live90"])
    e_motion = energy_share(r, vd, mot["live90"])
    e_off_motion = 1.0 - e_motion
    e_true_in_live = energy_share(true_d, vz, occ["live90"])
    e_true_in_dead = energy_share(true_d, vz, occ["dead"])
    e_pred_in_dead = energy_share(pred_d, vz, occ["dead"])

    majority = float(th["energy_majority"])
    if e_dead >= majority:
        where = "DEAD_LEAK"
        where_reason = "Majority of perpendicular energy is in occupancy-dead dims."
    elif e_motion >= majority:
        where = "MOTION_CONFUSION"
        where_reason = "Majority of perpendicular energy lies in the true-Δz motion span (wrong heading among real motions)."
    elif e_live >= majority:
        where = "LIVE_OFF_MOTION"
        where_reason = "Smear is in live occupancy dims but off the true-Δz motion span."
    else:
        where = "MIXED"
        where_reason = "No subspace holds a majority of perpendicular energy."

    knn = None
    r2_block = {}
    if actions is not None and step["path_state"] is not None:
        st = step["path_state"]
        st_t = st[:, HISTORY - 1 : -1].reshape(-1, st.shape[-1])
        act = np.asarray(actions, dtype=np.float64)
        if act.ndim == 3:
            # actions[pair, t] produces frame t+1; Δ at local t uses action HISTORY-1+t
            a_t = act[:, HISTORY - 1 : HISTORY - 1 + step["n_t"]].reshape(-1, act.shape[-1])
        else:
            a_t = act
        n = min(len(st_t), len(a_t), len(step["r"]))
        feat = np.concatenate([st_t[:n], a_t[:n]], axis=-1)
        knn = knn_neighbor_cosine(
            feat[:n],
            step["r"][:n],
            step["pair_id"][:n],
            step["t_id"][:n],
            step["mask"][:n],
            k=k,
            exclude_dt=excl,
        )
        dpose = pose_delta(st)[:n]
        train = (step["pair_id"][:n] % 2) == 0
        mm = step["mask"][:n]
        live_basis = vz[:, occ["live90"]]
        kw = dict(train=train[mm], basis=live_basis)
        r2_block = {
            "pose_from_true_dz": ridge_r2(step["true_d"][:n][mm], dpose[mm], **kw),
            "pose_from_pred_dz": ridge_r2(step["pred_d"][:n][mm], dpose[mm], **kw),
            "pose_from_perp_r": ridge_r2(step["r"][:n][mm], dpose[mm], **kw),
            "pose_from_parallel": ridge_r2((step["u"][:n] * step["par_len"][:n, None])[mm], dpose[mm], **kw),
            "r_from_state_action": ridge_r2(feat[:n][mm], step["r"][:n][mm], train=train[mm]),
            "pose_basis": "occupancy_live90",
        }

    sys_cut = float(th["neighbor_cosine_systematic_at_or_above"])
    rnd_cut = float(th["neighbor_cosine_random_below"])
    phys_cut = float(th["pose_from_perp_r2_physics_at_or_above"])
    junk_cut = float(th["pose_from_perp_r2_junk_below"])
    if knn is None:
        systematic = "NO_ACTIONS"
        sys_reason = "Need oracle actions for the neighbor test."
    elif knn["mean_cosine"] >= sys_cut:
        systematic = "SYSTEMATIC"
        sys_reason = "Perp residual aligns across nearby (state, action) pairs."
    elif knn["mean_cosine"] < rnd_cut:
        systematic = "RANDOM"
        sys_reason = "Perp residual is uncorrelated across nearby (state, action) pairs."
    else:
        systematic = "PARTIAL"
        sys_reason = "Neighbor cosine of r sits between the systematic and random cuts."

    r2r = r2_block.get("pose_from_perp_r")
    r2_true = r2_block.get("pose_from_true_dz")
    if r2_true is None or r2_true != r2_true or r2_true < junk_cut:
        physics = "CALIB_FAIL"
        phys_reason = (
            "Positive control failed: true Δz does not linearly predict sim Δpose "
            "on the held-out split. Do not read PERP_NO_POSE from this arm."
        )
    elif r2r is None or r2r != r2r:
        physics = "NO_POSE"
        phys_reason = "Could not fit held-out Δpose from the residual."
    elif r2r >= phys_cut:
        physics = "PERP_HAS_POSE"
        phys_reason = "Perpendicular residual still linearly predicts sim Δpose — smear is not pure junk."
    elif r2r < junk_cut:
        physics = "PERP_NO_POSE"
        phys_reason = "Perpendicular residual does not linearly predict sim Δpose (control passed)."
    else:
        physics = "PARTIAL"
        phys_reason = "Pose-from-residual R² is between the physics and junk cuts."

    return {
        "n_steps": int(m.sum()),
        "occupancy": {
            "k90": occ["k90"],
            "n_dead": occ["n_dead"],
            "dead_frac_dims": occ["dead_frac_dims"],
            "perp_energy_dead": e_dead,
            "perp_energy_live90": e_live,
            "true_energy_dead": e_true_in_dead,
            "true_energy_live90": e_true_in_live,
            "pred_energy_dead": e_pred_in_dead,
        },
        "motion": {
            "k90": mot["k90"],
            "perp_energy_motion90": e_motion,
            "perp_energy_off_motion": e_off_motion,
        },
        "knn": knn,
        "r2": r2_block,
        "decision": {
            "where": where,
            "where_reason": where_reason,
            "systematic": systematic,
            "systematic_reason": sys_reason,
            "physics": physics,
            "physics_reason": phys_reason,
            "cuts": {
                "occupancy_dead_rel": dead_rel,
                "occupancy_live_var_frac": live_frac,
                "motion_var_frac": motion_frac,
                "knn_k": k,
                "energy_majority": majority,
                "neighbor_cosine_systematic_at_or_above": sys_cut,
                "neighbor_cosine_random_below": rnd_cut,
                "pose_from_perp_r2_physics_at_or_above": phys_cut,
                "pose_from_perp_r2_junk_below": junk_cut,
            },
        },
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ca0", type=Path, required=True)
    p.add_argument("--oracle-bank", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", default="bank")
    args = p.parse_args(argv)
    th = load_thresholds().get("smear_structure") or {}
    ca0, _ = load_ca0(args.ca0)
    actions = None
    if args.oracle_bank is not None:
        blob = np.load(Path(args.oracle_bank) / "pairs.npz", allow_pickle=True)
        actions = blob["actions"]
    payload = analyze_bank(ca0, actions, th)
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.tag}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2))
    print(json.dumps(_jsonable(payload["decision"]), indent=2))
    print(json.dumps(_jsonable({k: payload[k] for k in ("n_steps", "occupancy", "motion", "knn", "r2")}), indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
