"""Simulated quantum annealing (SQA): path-integral Monte Carlo of the
transverse-field Ising model, a classical proxy of quantum-annealing dynamics.

The QUBO  E(x) = x^T Q x + H.x + const  (Q symmetric, zero diagonal) is
written in Ising form with s = 2x - 1,

    E = sum_{i<j} J_ij s_i s_j + sum_i hI_i s_i + const',
    J_ij = Q_ij / 2,   hI = H / 2 + (Q 1) / 2,

and divided by its largest |coefficient| M (per linear-term row), so that the
dimensionless problem Hamiltonian e(s) has coefficients in [-1, 1].  The
quantum annealing Hamiltonian  H(t) = e(s) - Gamma(t) sum_i sigma^x_i  at
inverse temperature beta is mapped by the Suzuki-Trotter decomposition onto
P coupled replicas (slices) s^1..s^P of the classical spins with the action

    S = (beta / P) sum_k e(s^k) - J_perp(t) sum_k sum_i s_i^k s_i^{k+1},
    J_perp = 1/2 ln coth(beta Gamma / P)          (periodic in k).

Gamma decreases linearly from ``gamma0`` to ``gamma1`` over ``sweeps``
Monte Carlo sweeps at fixed beta.  One sweep = single-spin Metropolis on
every spin of every slice (chromatic order: the colour classes of the
coupling graph, and within a class first the even, then the odd slices, so
that the spins updated together never interact -- an exact parallel
Metropolis scheme), followed, if ``global_moves``, by one Metropolis attempt
per spin to flip it in all slices at once (the inter-slice term is unchanged
by such a move).  The result per replica is the slice of lowest true energy;
``polish`` adds a zero-temperature steepest descent from it (classical
post-processing, as in the ``sa`` backend).  Several linear terms (rows of
``h``, shape (K, n)) are handled like in :func:`freeto.quantum.anneal.
simulated_annealing`: replicas are split evenly among the rows.

This is a classical Monte Carlo simulation of the equilibrium path integral
along the anneal, not a simulation of the unitary dynamics of a quantum
annealer and not quantum hardware.
"""
from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from .anneal import _Colored, _as_rows

__all__ = ["simulated_quantum_annealing", "SQA_DEFAULTS"]

try:  # in-place CSR x dense accumulation (Y += A X); optional fast path
    from scipy.sparse._sparsetools import csr_matvecs as _matvecs
except Exception:  # noqa: BLE001
    _matvecs = None

SQA_DEFAULTS = {"trotter": 16, "beta": 64.0, "gamma0": 1.0, "gamma1": 1e-3,
                "sweeps": 100, "global_moves": True, "polish": True}


def _descent(Qd, Hr, X, max_steps=None):
    """Steepest single-flip descent, replica r uses linear term Hr[r]."""
    R, n = X.shape
    X = X.copy()
    F = X @ Qd
    ar = np.arange(R)
    tol = -1e-14 * max(1.0, float(np.abs(Hr).max()))
    steps = 0
    for steps in range(1, (max_steps or 4 * n + 10) + 1):
        dE = (1.0 - 2.0 * X) * (Hr + 2.0 * F)
        j = np.argmin(dE, axis=1)
        go = dE[ar, j] < tol
        if not go.any():
            break
        r = ar[go]
        jj = j[go]
        d = 1.0 - 2.0 * X[r, jj]
        X[r, jj] += d
        F[r] += d[:, None] * Qd[jj]
    return X, steps


def simulated_quantum_annealing(Q, h, const=0.0, num_reads=32, sweeps=None, seed=None,
                                trotter=None, beta=None, gamma0=None, gamma1=None,
                                global_moves=None, polish=None, initial_state=None,
                                return_slices=False):
    """SQA (see module docs).  ``h``: (n,) or (K, n).

    Returns dict(samples (R, n) in {0,1}, energies (R,) true QUBO energies,
    row (R,), info)."""
    d = SQA_DEFAULTS
    P = int(trotter or d["trotter"])
    if P < 2 or P % 2:
        raise ValueError("trotter (number of slices P) must be an even number >= 2")
    beta = float(beta if beta is not None else d["beta"])
    g0 = float(gamma0 if gamma0 is not None else d["gamma0"])
    g1 = float(gamma1 if gamma1 is not None else d["gamma1"])
    sweeps = int(sweeps or d["sweeps"])
    global_moves = d["global_moves"] if global_moves is None else bool(global_moves)
    polish = d["polish"] if polish is None else bool(polish)
    rng = np.random.default_rng(seed)
    n = Q.shape[0]
    hrows = _as_rows(h, n)
    K = hrows.shape[0]
    R = max(int(num_reads), K)
    R = int(math.ceil(R / K) * K)
    row = np.repeat(np.arange(K), R // K)
    if n == 0:
        return dict(samples=np.zeros((R, 0)), energies=np.full(R, const), row=row,
                    info={"sweeps": 0})
    Qs = sp.csr_matrix(Q, dtype=np.float64)
    # Ising form, per-row normalisation
    Jfull = 0.5 * Qs                                   # J_ij for i != j (symmetric)
    hI = 0.5 * hrows + 0.5 * np.asarray(Qs.sum(axis=1)).ravel()[None, :]
    Jmax = float(abs(Jfull).max()) if Jfull.nnz else 0.0
    M = np.maximum(np.abs(hI).max(axis=1), Jmax)       # (K,)
    M = np.where(M > 0, M, 1.0)
    Mr = M[row]                                        # (R,)
    col = Qs.nnz > 0
    colored = _Colored(Qs) if col else None
    if colored is not None:
        perm, inv = colored.perm, colored.inv
        Jp = (0.5 * colored.Qp).tocsr()
        ranges = colored.ranges
        subs = [(rows_, 0.5 * S_) for rows_, S_ in colored.sub]
    else:
        perm = inv = np.arange(n)
        Jp = sp.csr_matrix((n, n))
        ranges = [(i, i + 1) for i in range(n)]
        subs = [(np.zeros(0, dtype=np.int64), sp.csr_matrix((0, 1))) for _ in range(n)]
    # normalised linear term per (variable, replica) in the working order
    hn = (hI[row].T / Mr[None, :])[perm]               # (n, R)
    # slices k = 2 j + p are stored as [p][variable][j, replica] so that each
    # parity class is one contiguous (n, P/2 * R) block; fields F = J S
    # (unnormalised J; divided by M per replica below)
    H2 = P // 2
    C = H2 * R
    if initial_state is not None:
        x0 = np.asarray(initial_state, dtype=np.float64).reshape(-1, n)
        S0 = np.empty((n, P, R))
        for r in range(R):
            S0[:, :, r] = (2.0 * x0[r % x0.shape[0]][perm] - 1.0)[:, None]
    else:
        S0 = np.where(rng.random((n, P, R)) < 0.5, -1.0, 1.0)
    S = [np.ascontiguousarray(S0[:, p::2, :].reshape(n, C)) for p in (0, 1)]
    hnC = np.tile(hn, (1, H2))                          # (n, C): column j*R + r -> hn[:, r]
    invMC = np.tile(1.0 / Mr, H2)[None, :]              # (1, C)
    # normalised fields Fn = (J S) / M (per replica column)
    Fn = [np.asarray(Jp @ S[p]) * invMC for p in (0, 1)]
    # coupling columns of every colour class (sparse, restricted to the rows
    # they reach)
    cols = []
    for (a, b), (rows_, Sub) in zip(ranges, subs):
        if rows_.size == 0:
            cols.append(None)
            continue
        Sc = sp.coo_matrix(Sub)
        A = sp.csr_matrix((Sc.data, (rows_[Sc.row], Sc.col)), shape=(n, b - a))
        A.sum_duplicates()
        A.sort_indices()
        A = sp.csr_matrix((A.data.astype(np.float64), A.indices.astype(np.int32),
                           A.indptr.astype(np.int32)), shape=A.shape)
        cols.append(A)
    # column maps of the neighbouring slices: parity 0 slice 2j has the odd
    # neighbours 2j-1 (odd index j-1) and 2j+1 (odd index j); parity 1 slice
    # 2j+1 has the even neighbours 2j (index j) and 2j+2 (index j+1)
    jj = np.repeat(np.arange(H2), R)
    rr = np.tile(np.arange(R), H2)
    nb_idx = (((jj - 1) % H2) * R + rr, ((jj + 1) % H2) * R + rr)
    bP = beta / P
    gam = np.linspace(g0, g1, sweeps) if sweeps > 1 else np.array([g1])
    n_acc = 0
    n_glob = 0
    ncol = len(ranges)

    def apply(c, p, a, b, dlt):
        A = cols[c]
        if A is None:
            return
        dl = np.ascontiguousarray(dlt * invMC)
        if _matvecs is not None:
            # in place: Fn[p] += A @ dl (no temporary)
            _matvecs(n, b - a, C, A.indptr, A.indices, A.data, dl.ravel(), Fn[p].ravel())
        else:
            Fn[p] += A @ dl

    for t in range(sweeps):
        x = beta * gam[t] / P
        jp = 0.5 * math.log(1.0 / math.tanh(x)) if x > 0 else 50.0
        thr = [-np.log(rng.random((n, C)) + 1e-300) for _ in (0, 1)]
        thr_g = -np.log(rng.random((n, R)) + 1e-300) if global_moves else None
        for c in rng.permutation(ncol):
            a, b = ranges[c]
            for p in (0, 1):
                s = S[p][a:b]
                o = S[1 - p][a:b]
                nb = o + o[:, nb_idx[p]]
                dS = s * (2.0 * jp * nb - 2.0 * bP * (hnC[a:b] + Fn[p][a:b]))
                acc = dS < thr[p][a:b]
                if acc.any():
                    dlt = np.where(acc, -2.0 * s, 0.0)
                    s += dlt
                    apply(c, p, a, b, dlt)
                    n_acc += int(np.count_nonzero(acc))
            if global_moves:
                g = (-2.0 * bP) * ((S[0][a:b] * (hnC[a:b] + Fn[0][a:b]))
                                   + (S[1][a:b] * (hnC[a:b] + Fn[1][a:b])))
                g = g.reshape(b - a, H2, R).sum(axis=1)
                acc = g < thr_g[a:b]
                if acc.any():
                    accC = np.tile(acc, (1, H2))
                    for p in (0, 1):
                        s = S[p][a:b]
                        dlt = np.where(accC, -2.0 * s, 0.0)
                        s += dlt
                        apply(c, p, a, b, dlt)
                    n_glob += int(np.count_nonzero(acc))
    S = np.stack([S[p].reshape(n, H2, R) for p in (0, 1)], axis=2).reshape(n, P, R)
    # back to the original order, x in {0,1}; true energies of every slice
    X = (0.5 * (S[inv] + 1.0))                                         # (n, P, R)
    Xs = X.transpose(2, 1, 0).reshape(R * P, n)                        # replica-major
    Hs = hrows[np.repeat(row, P)]
    QX = (Qs @ Xs.T).T
    Es = np.einsum("ij,ij->i", QX, Xs) + np.einsum("ij,ij->i", Xs, Hs) + const
    Es = Es.reshape(R, P)
    kbest = np.argmin(Es, axis=1)
    best = Xs.reshape(R, P, n)[np.arange(R), kbest]
    E = Es[np.arange(R), kbest]
    raw_E = E.copy()
    n_pol = 0
    if polish:
        Qd = Qs.toarray()
        best, n_pol = _descent(Qd, hrows[row], best)
        E = np.einsum("ij,ij->i", best @ Qd, best) + np.einsum("ij,ij->i", best, hrows[row]) \
            + const
    info = {"sweeps": sweeps, "trotter": P, "beta": beta, "gamma0": g0, "gamma1": g1,
            "global_moves": global_moves, "polish": polish, "n_colors": ncol,
            "accepted_local": n_acc, "accepted_global": n_glob, "polish_steps": n_pol,
            "raw_energies": raw_E}
    out = dict(samples=np.ascontiguousarray(best), energies=E, row=row, info=info)
    if return_slices:
        out["slice_samples"] = Xs.reshape(R, P, n)
        out["slice_energies"] = Es
    return out
