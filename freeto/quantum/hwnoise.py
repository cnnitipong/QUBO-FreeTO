"""Analogue-hardware error models for the QUBO solves (perturbed coefficients).

A quantum annealer realises the Ising coefficients h_i, J_ij of the problem
it is given only approximately: they are scaled to the device range and set
with limited precision and with control errors.  This module perturbs the
coefficients of a QUBO before it is handed to a solver, so that the solver
minimises the perturbed problem while everything else (selection of the best
sample, volume control, repair, acceptance) keeps using the true model.

Activated through ``QUBOOptions.backend_options["perturb"]``, a dict with

* ``"noise"``: relative Gaussian noise sigma; every nonzero coefficient c of
  the Ising form becomes c (1 + sigma xi), xi ~ N(0, 1) independent;
* ``"bits"``: quantisation to b bits (sign included): with M the largest
  |coefficient| (h and J) of the problem passed to the solver in one call,
  c -> M/L round(c L / M), L = 2^(b-1) - 1 levels per sign (coefficients
  below M/(2L) become 0);
* ``"seed"`` (optional): seed of the perturbation stream (default: the
  QUBO seed).

Both may be combined (noise first, then quantisation).  Conventions:

* Ising form with s = 2x - 1 in {-1, +1}:
  x^T Q x + H.x = sum_{i<j} J_ij s_i s_j + sum_i hI_i s_i + const with
  J_ij = Q_ij / 2 and hI = H / 2 + (Q 1) / 2 (Q symmetric, zero diagonal);
* a call with several linear terms (rows of H, the volume multipliers of the
  multisection) shares one coupling matrix: the J noise is drawn once per
  call, the h noise independently for every row, and the quantisation scale
  M is the largest |coefficient| over all rows of the call;
* the solver's samples are re-scored with the true model; the caller keeps the
  sample of lowest true energy (re-evaluation of returned samples with the
  submitted model, as done with hardware samples).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["qubo_to_ising", "ising_to_qubo", "perturb_qubo", "quantize",
           "bits_for_relative_accuracy", "PERTURB_KEYS"]

PERTURB_KEYS = ("noise", "bits", "seed")


def _rows(H, n):
    H = np.asarray(H, dtype=np.float64)
    return (H[None, :] if H.ndim == 1 else H), H.ndim == 1


def qubo_to_ising(Q, H):
    """(Q sym zero-diag, H (n,) or (K, n)) -> (Ju upper-triangle COO (i<j),
    hI (K, n)).  ``Ju.data`` are the pair couplings J_ij (each pair once)."""
    Qs = sp.csr_matrix(Q)
    n = Qs.shape[0]
    Hr, _ = _rows(H, n)
    Ju = sp.triu(Qs, k=1).tocoo()
    Ju = sp.coo_matrix((0.5 * Ju.data, (Ju.row, Ju.col)), shape=(n, n))
    rs = np.asarray(Qs.sum(axis=1)).ravel()
    hI = 0.5 * Hr + 0.5 * rs[None, :]
    return Ju, hI


def ising_to_qubo(Ju, hI):
    """Inverse of :func:`qubo_to_ising` (const dropped).  Returns (Q CSR
    symmetric zero diagonal, H (K, n))."""
    n = Ju.shape[0]
    U = sp.coo_matrix((2.0 * Ju.data, (Ju.row, Ju.col)), shape=(n, n))
    Q = sp.csr_matrix(U + U.T)
    rs = np.asarray(Q.sum(axis=1)).ravel()
    H = 2.0 * np.asarray(hI) - rs[None, :]
    return Q, H


def quantize(c, M, bits):
    """Round ``c`` to the b-bit grid of step M / (2^(b-1) - 1)."""
    L = float(2 ** (int(bits) - 1) - 1)
    if not M > 0:
        return np.zeros_like(c)
    return np.round(np.asarray(c) * (L / M)) * (M / L)


def bits_for_relative_accuracy(dynamic_range, rel=0.01):
    """Smallest b (sign included) such that the uniform b-bit grid scaled to
    the largest |coefficient| represents the smallest nonzero |coefficient|
    (max/min = ``dynamic_range``) to the relative accuracy ``rel`` (rounding
    error <= half a step):  (M / L) / 2 <= rel * M / DR  <=>
    L >= DR / (2 rel),  b = 1 + ceil(log2(DR / (2 rel) + 1))."""
    dr = float(dynamic_range)
    if not np.isfinite(dr):
        return float("inf")
    return int(1 + np.ceil(np.log2(dr / (2.0 * rel) + 1.0)))


def perturb_qubo(Q, H, spec, rng):
    """Perturbed copy of the QUBO (see module docs).

    Returns (Qp CSR, Hp with the shape of ``H``, info dict)."""
    n = Q.shape[0]
    Hr, one = _rows(H, n)
    Ju, hI = qubo_to_ising(Q, Hr)
    J = Ju.data.copy()
    hh = hI.copy()
    sig = float(spec.get("noise") or 0.0)
    if sig > 0:
        J = J * (1.0 + sig * rng.standard_normal(J.size))
        hh = hh * (1.0 + sig * rng.standard_normal(hh.shape))
    info = {"noise": sig, "bits": spec.get("bits")}
    if spec.get("bits"):
        b = int(spec["bits"])
        M = max(float(np.abs(J).max(initial=0.0)), float(np.abs(hh).max(initial=0.0)))
        J = quantize(J, M, b)
        hh = quantize(hh, M, b)
        info["scale"] = M
        info["J_zeroed"] = float(np.mean(J == 0)) if J.size else 0.0
        info["h_zeroed"] = float(np.mean(hh == 0)) if hh.size else 0.0
    Jp = sp.coo_matrix((J, (Ju.row, Ju.col)), shape=(n, n))
    Qp, Hp = ising_to_qubo(Jp, hh)
    Qp.eliminate_zeros()
    return Qp, (Hp[0] if one else Hp), info
