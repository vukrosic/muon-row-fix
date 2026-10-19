"""Exact Muon step length per neuron at two real layer shapes (no training).
tall: Qwen3.8 27B MLP up-projection, 17408 x 5120.  wide: DeepSeek-V4 Pro expert up-projection, 3072 x 7168.
Each neuron's gradient row = random direction x size s (lognormal, sigma 0.5). Muon's exact update O = msign(M).
Prints the log-log slope of step length on gradient size (predicted 1 - n/m for tall, 0 for wide).
usage: python3 realshape.py charts/realshape.npz    (about 1 minute on a laptop CPU; then python3 charts/chart_real.py charts/realshape.npz)
"""
import numpy as np, time, sys
import scipy.linalg as sl
np.seterr(all="ignore")   # macOS Accelerate raises false matmul warnings
rng = np.random.default_rng(0)
out = {}
for name, m, n in [("tall", 17408, 5120), ("wide", 3072, 7168)]:
    t = time.time()
    s = np.exp(rng.normal(0, 0.5, m)); s /= s.mean()          # gradient size per neuron
    Z = rng.standard_normal((m, n)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    M = s[:, None] * Z
    if m > n:   # O = M (MᵀM)^-1/2 ; |row i|² = m_iᵀ (MᵀM)⁻¹ m_i
        A = M.T @ M
        L = np.linalg.cholesky(A)
        X = sl.solve_triangular(L, M.T, lower=True)   # L⁻¹ m_i
        ell = np.sqrt((X * X).sum(0))
    else:       # O = (MMᵀ)^-1/2 M
        w, V = np.linalg.eigh(M @ M.T)
        O = (V * w**-0.5) @ V.T @ M
        ell = np.linalg.norm(O, axis=1)
    rel = ell / np.sqrt((ell**2).mean())
    k = np.polyfit(np.log(s), np.log(rel), 1)[0]
    print(name, m, n, f"{time.time()-t:.1f}s  slope={k:.3f}  pred={max(0,1-n/m):.3f}  step range {rel.min():.3f}-{rel.max():.3f}  CV={rel.std()/rel.mean():.3f}")
    out[name + "_s"] = s; out[name + "_l"] = rel
np.savez(sys.argv[1] if len(sys.argv) > 1 else "realshape.npz", **out)
