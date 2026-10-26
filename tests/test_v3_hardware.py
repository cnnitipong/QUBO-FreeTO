"""v3 additions: coefficient perturbation (analogue-hardware error model,
freeto.quantum.hwnoise) and the simulated quantum annealing backend
(freeto.quantum.sqa, backend "sqa")."""
import math
import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from freeto import example_config, run_freeto                                  # noqa: E402
from freeto.quantum import solve_qubo                                          # noqa: E402
from freeto.quantum import backends as qb                                      # noqa: E402
from freeto.quantum.blockqubo import solve_volume_model                        # noqa: E402
from freeto.quantum.exact import exact_solve                                   # noqa: E402
from freeto.quantum.hwnoise import (bits_for_relative_accuracy, ising_to_qubo,  # noqa: E402
                                    perturb_qubo, quantize, qubo_to_ising)
from freeto.quantum.qubo import energy                                         # noqa: E402
from freeto.quantum.sqa import simulated_quantum_annealing                     # noqa: E402


def rand_qubo(n, rng, density=1.0):
    A = rng.normal(size=(n, n))
    A = np.triu(A, 1) * (rng.random((n, n)) < density)
    Q = A + A.T
    return Q, 2.0 * rng.normal(size=n)


# ---------------------------------------------------------------------------
# hwnoise
# ---------------------------------------------------------------------------
def test_ising_roundtrip_and_energy_identity():
    rng = np.random.default_rng(0)
    Q, h = rand_qubo(9, rng, 0.6)
    H = np.vstack([h, h + 1.0])
    Ju, hI = qubo_to_ising(sp.csr_matrix(Q), H)
    Q2, H2 = ising_to_qubo(Ju, hI)
    assert np.allclose(Q2.toarray(), Q) and np.allclose(H2, H)
    X = (rng.random((20, 9)) < 0.5).astype(float)
    S = 2 * X - 1
    Jd = Ju.toarray()
    for k in range(2):
        Eq = energy(Q, H[k], 0.0, X)
        Ei = np.einsum("ri,ij,rj->r", S, Jd, S) + S @ hI[k]
        d = Eq - Ei
        assert np.allclose(d, d[0])                  # equal up to a constant


def test_perturb_identity_quantize_and_noise_statistics():
    rng = np.random.default_rng(1)
    Q, h = rand_qubo(12, rng, 0.7)
    Qs = sp.csr_matrix(Q)
    Qp, hp, _ = perturb_qubo(Qs, h, {}, np.random.default_rng(0))
    assert np.allclose(Qp.toarray(), Q) and np.allclose(hp, h)
    # quantisation: coefficients on the grid, small ones zeroed
    c = np.array([1.0, 0.5, 0.26, 0.01, -0.7])
    q = quantize(c, 1.0, 3)                          # L = 3 levels per sign
    assert np.allclose(q * 3, np.round(q * 3)) and q[3] == 0.0
    Qp, hp, info = perturb_qubo(Qs, h, {"bits": 4}, np.random.default_rng(0))
    Ju, hI = qubo_to_ising(Qp, hp)
    M = info["scale"]
    L = 7
    assert np.allclose(Ju.data * L / M, np.round(Ju.data * L / M))
    assert np.allclose(hI * L / M, np.round(hI * L / M))
    # relative noise: (c' - c) / c ~ N(0, sigma)
    rs = []
    J0, _ = qubo_to_ising(Qs, h)
    for s in range(30):
        Qn, _, _ = perturb_qubo(Qs, h, {"noise": 0.05}, np.random.default_rng(s))
        Jn = 0.5 * Qn.toarray()[J0.row, J0.col]
        rs.append(Jn / J0.data - 1.0)
    rs = np.concatenate(rs)
    assert abs(rs.mean()) < 0.01 and abs(rs.std() - 0.05) < 0.01


def test_bits_for_relative_accuracy():
    assert bits_for_relative_accuracy(1.0) == 1 + math.ceil(math.log2(51))
    # b bits represent DR to 1 %: half a step <= 1 % of the smallest coefficient
    for dr in (3.0, 47.0, 1e3, 1e5):
        b = bits_for_relative_accuracy(dr)
        L = 2 ** (b - 1) - 1
        assert 0.5 / L <= 0.01 / dr + 1e-15
        L2 = 2 ** (b - 2) - 1
        assert 0.5 / max(L2, 1) > 0.01 / dr


def test_volume_model_with_perturbation_is_feasible_and_rescored():
    rng = np.random.default_rng(2)
    m = 40
    Q, h = rand_qubo(m, rng, 0.2)
    Q = sp.csr_matrix(0.2 * Q)
    v = rng.uniform(0.5, 1.5, m)
    for spec in ({"noise": 0.1}, {"bits": 4}, {"noise": 0.02, "bits": 6}):
        y, info = solve_volume_model(Q, h, 0.0, v, 0.0, 0.4 * v.sum(), np.ones(m),
                                     backend="sa", seed=3, backend_opts={"perturb": spec,
                                                                         "sweeps": 50})
        assert set(np.unique(y)) <= {0.0, 1.0}
        assert v @ y <= 0.4 * v.sum() + 1e-9
        assert np.isclose(info["energy"], energy(Q, h, 0.0, y))
    # block path (several blocks, sequential bisection)
    blocks = [np.arange(0, 20), np.arange(20, 40)]
    y, info = solve_volume_model(Q, h, 0.0, v, 0.0, 0.4 * v.sum(), np.ones(m),
                                 backend="sa", blocks=blocks, seed=3,
                                 backend_opts={"perturb": {"noise": 0.05}})
    assert v @ y <= 0.4 * v.sum() + 1e-9


def test_perturbation_option_through_updater_is_deterministic():
    kw = dict(mesh_control=20, optimizer="QUBO", max_iter=6)
    q = {"backend": "sa", "hessian": "block", "seed": 0,
         "backend_options": {"perturb": {"noise": 0.05}}}
    r1 = run_freeto(example_config("cantilever_beam", qubo=dict(q), **kw), log=None)
    r2 = run_freeto(example_config("cantilever_beam", qubo=dict(q), **kw), log=None)
    assert r1.comp == r2.comp and math.isfinite(r1.comp)
    r0 = run_freeto(example_config("cantilever_beam", qubo={"backend": "sa", "hessian": "block",
                                                             "seed": 0}, **kw), log=None)
    assert math.isfinite(r0.comp)


# ---------------------------------------------------------------------------
# SQA
# ---------------------------------------------------------------------------
def test_sqa_registered_and_batch_capable():
    assert "sqa" in qb.BACKENDS and qb.is_batch_backend("sqa")
    assert qb.check_backend("sqa")


@pytest.mark.parametrize("n", [8, 11, 14])
def test_sqa_finds_exact_optimum_on_small_random_qubos(n):
    rng = np.random.default_rng(n)
    hits = 0
    for t in range(6):
        Q, h = rand_qubo(n, rng, 0.8)
        _, Eex, _ = exact_solve(Q, h, 0.0)
        r = solve_qubo(Q, h, 0.0, backend="sqa", num_reads=4, seed=t, sweeps=60)
        assert r.energy >= Eex - 1e-9
        hits += r.energy <= Eex + 1e-9 * max(1.0, abs(Eex))
    assert hits >= 5


def test_sqa_raw_without_polish_and_slices():
    rng = np.random.default_rng(5)
    Q, h = rand_qubo(10, rng, 0.8)
    _, Eex, _ = exact_solve(Q, h, 0.0)
    r = simulated_quantum_annealing(sp.csr_matrix(Q), h, num_reads=4, seed=0, polish=False,
                                    return_slices=True)
    assert r["slice_samples"].shape == (4, 16, 10)
    E = r["slice_energies"]
    assert np.allclose(r["energies"], E.min(axis=1))
    assert r["energies"].min() <= Eex + 1e-9 * max(1.0, abs(Eex)) or \
        r["energies"].min() - Eex < 0.05 * abs(Eex)
    # energies are true QUBO energies of the returned samples
    assert np.allclose(r["energies"], energy(Q, h, 0.0, r["samples"]))


def test_sqa_rows_reproducible_and_validation():
    rng = np.random.default_rng(6)
    Q, h = rand_qubo(30, rng, 0.15)
    H = np.vstack([h, h + 2.0, h - 2.0])
    a = simulated_quantum_annealing(sp.csr_matrix(Q), H, num_reads=3, seed=4, sweeps=30)
    b = simulated_quantum_annealing(sp.csr_matrix(Q), H, num_reads=3, seed=4, sweeps=30)
    assert np.array_equal(a["samples"], b["samples"])
    assert list(a["row"]) == [0, 1, 2]
    for k in range(3):
        assert np.isclose(a["energies"][k], energy(Q, H[k], 0.0, a["samples"][k]))
    with pytest.raises(ValueError):
        simulated_quantum_annealing(sp.csr_matrix(Q), h, trotter=5)
    # no couplings: separable problem solved exactly
    r = simulated_quantum_annealing(sp.csr_matrix((5, 5)), np.array([1., -1, 2, -3, 0.5]),
                                    num_reads=2, seed=0)
    assert np.array_equal(r["samples"][0], [0, 1, 0, 1, 0])


def test_sqa_in_volume_model_multisection():
    rng = np.random.default_rng(7)
    m = 50
    Q, h = rand_qubo(m, rng, 0.1)
    Q = sp.csr_matrix(0.3 * Q)
    v = rng.uniform(0.5, 1.5, m)
    y, info = solve_volume_model(Q, h, 0.0, v, 0.0, 0.5 * v.sum(), np.ones(m), backend="sqa",
                                 seed=1, backend_opts={"sweeps": 30})
    assert v @ y <= 0.5 * v.sum() + 1e-9
    assert info["stats"]["n_solves"] >= 1 and info["backend"] == "sqa"


def test_sqa_continuum_end_to_end_short():
    cfg = example_config("cantilever_beam", mesh_control=20, optimizer="QUBO", max_iter=5,
                         qubo={"backend": "sqa", "hessian": "block", "seed": 0,
                               "backend_options": {"sweeps": 20}})
    res = run_freeto(cfg, log=None)
    qh = res.extra["qubo_history"]
    assert qh and qh[-1]["backend"] == "sqa"
    assert math.isfinite(res.comp)
