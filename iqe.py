"""Interval Quasimetric Embeddings — IQE-sum (Wang & Isola, arXiv:2211.15120).

Implements Eqs. (2)–(3): reshape φ ∈ R^{k·l} → u ∈ R^{k×l}, then

  d_i(u,v) = | ⋃_j [u_{ij}, max(u_{ij}, v_{ij})] |
  d_IQE-sum(u,v) = Σ_i d_i(u,v)

The per-component measure follows the official torch-quasimetric reference
(https://github.com/quasimetric-learning/torch-quasimetric ``torchqmet.iqe``).
"""

from __future__ import annotations

import torch


def reshape_phi(phi: torch.Tensor, *, k: int, l: int) -> torch.Tensor:
    """Reshape last dim d=k*l into (..., k, l)."""
    d = phi.size(-1)
    if d != k * l:
        raise ValueError(f"phi last dim {d} != k*l={k * l}")
    return phi.unflatten(-1, (k, l))


@torch.jit.script
def _iqe_components(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Lebesgue measure of union of intervals per component.

    x, y: (..., k, l)  — last dim is dim_per_component.
    returns: (..., k)
    """
    D = x.shape[-1]

    valid = x < y

    xy = torch.cat(torch.broadcast_tensors(x, y), dim=-1)
    sxy, ixy = xy.sort(dim=-1)

    neg_inc_copies = torch.gather(valid, dim=-1, index=ixy % D) * torch.where(
        ixy < D, -1, 1
    )
    neg_inp_copies = torch.cumsum(neg_inc_copies, dim=-1)
    neg_f = (neg_inp_copies < 0) * (-1.0)
    neg_incf = torch.cat(
        [neg_f.narrow(-1, 0, 1), torch.diff(neg_f, dim=-1)], dim=-1
    )
    return (sxy * neg_incf).sum(-1)


def iqe_sum(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    k: int | None = None,
    l: int | None = None,
) -> torch.Tensor:
    """IQE-sum quasimetric between embeddings.

    Accepts either flat (..., d) with d=k*l (k,l required) or already
    reshaped (..., k, l). Returns nonnegative scalar per leading batch dim.
    """
    if u.ndim >= 2 and u.shape[-1] != v.shape[-1]:
        raise ValueError(f"mismatched last dims {u.shape[-1]} vs {v.shape[-1]}")

    if k is not None or l is not None:
        if k is None or l is None:
            raise ValueError("provide both k and l, or neither")
        u = reshape_phi(u, k=k, l=l)
        v = reshape_phi(v, k=k, l=l)
    elif u.ndim < 2:
        raise ValueError("flat vectors require k and l")

    # Ensure 2D component layout on last two dims
    if u.ndim == 1:
        raise ValueError("need at least (k, l) after reshape")
    comps = _iqe_components(u, v)
    return comps.sum(dim=-1)
