#!/usr/bin/env python3
"""CA3: intervention magnitude sweep (GPU). Fig-6 only plots the JSON.

  python scripts/intervene_sweep.py \\
      --dump eval_results/pusht/phase_b_dump/seed0/dump.npz \\
      --factor block_x --eps-range -3 3 --device cuda \\
      --out eval_results/pusht/ca3_sweep/seed0/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("STABLEWM_HOME", str(ROOT.parent / "stablewm"))

from eval_live import ENV_REGISTRY  # noqa: E402
from eval_setup import load_lewm_checkpoint  # noqa: E402
from phase_b import HISTORY, contact_events, load_dump  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from latent_probe import fit_linear_direction  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--factor", default="block_x")
    p.add_argument("--eps-range", type=float, nargs=2, default=[-3.0, 3.0])
    p.add_argument("--n-eps", type=int, default=13)
    p.add_argument("--n-base", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    dump = load_dump(args.dump)
    meta = dump.get("meta") or {}
    env = str(meta.get("env", "pusht"))
    names = [str(x) for x in np.asarray(dump["factor_names"]).tolist()]
    factor = args.factor if args.factor in names else ("block_x" if "block_x" in names else names[0])
    fi = names.index(factor)
    z = dump["z"]
    state = dump["state"]
    n, l, d = z.shape
    zf = z.reshape(n * l, d)
    y = state.reshape(n * l, state.shape[-1])[:, fi]
    direction = fit_linear_direction(zf, y)
    ev = contact_events(state, env=env)
    any_ev = ev["any"]
    if any_ev.ndim == 2:
        pred = np.zeros_like(any_ev, dtype=bool)
        if any_ev.shape[1] > HISTORY:
            pred[:, HISTORY:] = True
        else:
            pred[:, :] = True
        free_idx = list(zip(*np.where((~any_ev) & pred)))
        con_idx = list(zip(*np.where(any_ev & pred)))
    else:
        hist = HISTORY - 1
        free_idx = [(int(i), hist) for i in np.where(~any_ev)[0]]
        con_idx = [(int(i), hist) for i in np.where(any_ev)[0]]
    rng = np.random.default_rng(0)

    def _sample(idxs, k):
        if not idxs:
            return []
        pick = rng.choice(len(idxs), size=min(k, len(idxs)), replace=False)
        return [idxs[int(i)] for i in pick]

    free_b = _sample(free_idx, args.n_base)
    con_b = _sample(con_idx, args.n_base)
    if not con_b:
        con_b = free_b[:1]
    if not free_b:
        free_b = con_b[:1]

    spec = ENV_REGISTRY[env]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_lewm_checkpoint(spec.ckpt_dir)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    w = torch.from_numpy(direction.astype(np.float32)).to(device)
    eps_grid = np.linspace(args.eps_range[0], args.eps_range[1], args.n_eps)

    def _curve(bases):
        out = []
        with torch.no_grad():
            for eps in eps_grid:
                deltas = []
                for si, ti in bases:
                    t = int(ti)
                    t = min(max(t, HISTORY - 1), l - 1)
                    hist_z = torch.from_numpy(z[int(si), t - HISTORY + 1 : t + 1]).unsqueeze(0).to(device)
                    if hist_z.shape[1] < HISTORY:
                        continue
                    act = torch.zeros(1, HISTORY, 10, device=device)
                    act_emb = model.action_encoder(act)
                    pred0 = model.predict(hist_z, act_emb)[:, -1, :]
                    hist_p = hist_z.clone()
                    hist_p[:, -1, :] = hist_p[:, -1, :] + float(eps) * w
                    pred1 = model.predict(hist_p, act_emb)[:, -1, :]
                    d0 = float((pred0[0] * w).sum())
                    d1 = float((pred1[0] * w).sum())
                    deltas.append(d1 - d0)
                out.append(float(np.mean(deltas)) if deltas else float("nan"))
        return out

    payload = {
        "factor": factor,
        "eps": eps_grid.tolist(),
        "free": _curve(free_b),
        "contact": _curve(con_b),
        "n_free_bases": len(free_b),
        "n_contact_bases": len(con_b),
        "dump": str(args.dump),
        "note": "Δ = probe(P(z+εd)) − probe(P(z)); one P step, zero actions.",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "sweep.json"
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("factor", "n_free_bases", "n_contact_bases")}, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
