"""row_math_checks.py: tiny exact numerical checks for row-math-notes.md (no training, numpy float64, CPU).
usage: taskpolicy -b nice -n 20 python3 row_math_checks.py SECTION   (SECTION = c1 ... c16 or c14b, c14c, c14d; each under ~25 s)
Every number quoted in row-math-notes.md comes from one of these sections.
c13 (b) and c15 read momentum snapshots (snaps/*.pt) from small training runs that are not in this repo; c6 skips them when absent.
"""
import sys, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
np.seterr(all="ignore")                        # macOS Accelerate matmul raises false overflow warnings; results are checked below


def polar(M):                                   # msign(M) = U V^T by exact SVD
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def lev(M):                                     # leverage h_i = m_i^T (M^T M)^-1 m_i, rows of a tall M
    return np.einsum("ij,ij->i", M @ np.linalg.inv(M.T @ M), M)


def lognormal_rows(m, n, sig, rng):             # rows = s_i * (random direction), log s_i ~ N(0, sig^2)
    Z = rng.standard_normal((m, n)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return np.exp(sig * rng.standard_normal(m))[:, None] * Z


def formula_h(s2, n):                           # h_i = c s_i^2 / (1 + c s_i^2), c solves sum h_i = n (bisection in log c)
    lo, hi = -60.0, 60.0
    for _ in range(200):
        c = math.exp((lo + hi) / 2); t = (c * s2 / (1 + c * s2)).sum()
        lo, hi = ((lo + hi) / 2, hi) if t < n else (lo, (lo + hi) / 2)
    return c * s2 / (1 + c * s2)


def cv(v): return v.std() / v.mean()


def roweq(O):                                   # every row the same length, |O|_F kept (uni_opt.py "roweq")
    D = O / np.sqrt((O ** 2).mean(1, keepdims=True))
    return D * (np.linalg.norm(O) / np.linalg.norm(D))


def c1():
    print("C1 exact identities: |row i of msign(M)|^2 = h_i, sum h = n, Sherman-Morrison h = q/(1+q)")
    for m, n in [(512, 128), (2048, 512)]:
        M = lognormal_rows(m, n, 1.0, rng); O = polar(M); h = lev(M)
        A = M.T @ M; q = np.array([M[i] @ np.linalg.solve(A - np.outer(M[i], M[i]), M[i]) for i in range(5)])
        print(f"  {m}x{n}: max|row^2 - h| {np.abs((O ** 2).sum(1) - h).max():.1e}  sum h {h.sum():.6f}  "
              f"max|q/(1+q) - h| (5 rows) {np.abs(q / (1 + q) - h[:5]).max():.1e}  |O^T O - I| {np.abs(O.T @ O - np.eye(n)).max():.1e}  "
              f"min/max h {h.min():.4f} {h.max():.4f}")


def c2():
    print("C2 exact elasticity: scale row i of M by t, d log|row i of msign| / d log t = 1 - h_i")
    M = lognormal_rows(512, 128, 1.0, rng); h = lev(M); L = np.linalg.norm(polar(M), axis=1)
    idx = [int(np.argmin(h)), int(np.argsort(h)[256]), int(np.argmax(h))]
    for i in idx:
        eps = 1e-5; M2 = M.copy(); M2[i] *= math.exp(eps)
        L2 = np.linalg.norm(polar(M2), axis=1); d = (math.log(L2[i]) - math.log(L[i])) / eps
        others = (np.log(L2) - np.log(L)) / eps; others[i] = 0
        print(f"  row h={h[i]:.4f}: numeric {d:.5f}  1-h {1 - h[i]:.5f}   other rows: sum d(h_j) {((L2**2 - L**2)[np.arange(512) != i]).sum()/eps:+.5f} "
              f"vs -2h(1-h) {-2*h[i]*(1-h[i]):+.5f}")


def c3():
    print("C3 Gaussian M (no persistent structure): h_i ~ Beta(n/2, (m-n)/2); CV(h) = sqrt(2(m-n)/(n(m+2)))")
    for m, n in [(256, 64), (1024, 256), (2048, 512), (512, 256), (2048, 256)]:
        hs = np.concatenate([lev(rng.standard_normal((m, n))) for _ in range(4 if m * n < 3e5 else 1)])
        L = np.sqrt(hs)
        print(f"  {m}x{n} a={m/n:.0f}: mean h {hs.mean():.5f} (n/m {n/m:.5f})  CV(h) {cv(hs):.4f} vs {math.sqrt(2*(m-n)/(n*(m+2))):.4f}"
              f"  row-length CV {cv(L):.4f} vs ~sqrt((1-n/m)/(2n)) {math.sqrt((1-n/m)/(2*n)):.4f}")


def c4():
    print("C4 row-length CV of msign on tall M: formula vs exact SVD vs small-spread rule (1 - n/m) * sigma")
    n = 256
    for a in [1.5, 2, 2.7, 4, 8]:
        out = []
        for sig in [0.25, 0.5, 1.0]:
            m = int(a * n); M = lognormal_rows(m, n, sig, rng)
            ex = cv(np.linalg.norm(polar(M), axis=1))
            s2 = (M ** 2).sum(1); fo = cv(np.sqrt(formula_h(s2, n)))
            out.append(f"s{sig}: exact {ex:.3f} formula {fo:.3f} rule {(1 - 1/a)*sig:.3f}")
        print(f"  a={a}: " + " | ".join(out))


def c5():
    print("C5 direction matters: equal row norms; half the rows share one common direction b (weight w), half random")
    m, n = 1024, 256
    for w in [0.0, 0.5, 0.8]:
        Z = rng.standard_normal((m, n)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
        b = rng.standard_normal(n); b /= np.linalg.norm(b)
        Z[:m // 2] = w * b + math.sqrt(1 - w * w) * Z[:m // 2]
        Z /= np.linalg.norm(Z, axis=1, keepdims=True); h = lev(Z)
        print(f"  w={w}: mean h shared-direction rows {h[:m//2].mean():.4f}  other rows {h[m//2:].mean():.4f}  (n/m {n/m:.4f})")
    print("  spike in one row's direction: one row points along a direction no other row uses")
    Z = rng.standard_normal((m, n)); Z[:, 0] *= 0.1; Z[0] = 0; Z[0, 0] = math.sqrt(n)   # row 0 = same norm as the others, rare direction
    Z /= np.linalg.norm(Z, axis=1, keepdims=True); h = lev(Z)
    print(f"  rare-direction row h {h[0]:.4f} vs median {np.median(h):.4f}")


def c6():
    print("C6 equal-rows (roweq) geometry: cos(O, R) = 1/sqrt(1+CV^2) exactly; singular values of R in [l_rms/l_max, l_rms/l_min]")
    rows = []
    for a, sig in [(4, 0.25), (4, 0.5), (4, 1.0), (2, 1.0), (8, 1.0)]:
        m, n = int(a * 128), 128; rows.append((f"lognormal a={a} sig={sig}", lognormal_rows(m, n, sig, rng)))
    import torch
    for tag in ["mlpx_relu2", "mlpx_relu3", "mlpx_softplus", "mlpx_silu", "mlpx_gelu", "mlpx_relu", "mlpx_tanh", "mlpx_swiglu", "lm_swiglu"]:
        p = os.path.join(HERE, "snaps", tag + ".pt")
        if os.path.exists(p):
            d = torch.load(p, weights_only=False)
            k = max(d, key=lambda k: d[k].numel()); rows.append((f"{tag} {k} {tuple(d[k].shape)}", d[k].double().numpy()))
    for name, M in rows:
        O = polar(M); R = roweq(O); L = np.linalg.norm(O, axis=1); c = cv(L)
        cosv = (O * R).sum() / (np.linalg.norm(O) * np.linalg.norm(R)); sv = np.linalg.svd(R, compute_uv=False)
        lr = math.sqrt((L ** 2).mean()); n = M.shape[1]
        print(f"  {name}: CV {c:.3f}  cos {cosv:.5f} vs {1/math.sqrt(1+c*c):.5f}  sv(R) [{sv.min():.3f}, {sv.max():.3f}] "
              f"bound [{lr/L.max():.3f}, {lr/L.min():.3f}]  |R^TR-I|_F/sqrt(n) {np.linalg.norm(R.T @ R - np.eye(n))/math.sqrt(n):.3f}  "
              f"log-spread sig {np.log(np.linalg.norm(M, axis=1)).std():.2f}")


def c7():
    print("C7 wide M: rows of msign exactly unit length; columns carry the leverage. MoE shapes (scaled 1/16): fine-grained expert")
    for name, (m, n) in [("expert up (wide, d_ff<d_model) 128x448", (128, 448)), ("expert down (tall) 448x128", (448, 128)),
                         ("dense FFN up (tall) 512x128", (512, 128)), ("dense FFN down (wide) 128x512", (128, 512))]:
        M = lognormal_rows(m, n, 1.0, rng) if m > n else lognormal_rows(n, m, 1.0, rng).T   # hidden-neuron spread sig=1 on the hidden side
        O = polar(M); r = np.linalg.norm(O, axis=1); c = np.linalg.norm(O, axis=0)
        print(f"  {name}: row lengths CV {cv(r):.4f} (min {r.min():.4f} max {r.max():.4f})  column lengths CV {cv(c):.4f} (min {c.min():.4f} max {c.max():.4f})")


def c8():
    print("C8 activation exponents by quadrature, preactivation z = r*xi, xi ~ N(0,1), dense grid (40001 points on [-10, 10])")
    xi = np.linspace(-10, 10, 40001); w = np.exp(-xi ** 2 / 2); w = w / w.sum()
    Phi = lambda z: 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    sg = lambda z: 1 / (1 + np.exp(-z))
    acts = {
        "relu": (lambda z: np.maximum(z, 0), lambda z: (z > 0) * 1.0),
        "relu1.5": (lambda z: np.maximum(z, 0) ** 1.5, lambda z: 1.5 * np.maximum(z, 0) ** 0.5),
        "relu2": (lambda z: np.maximum(z, 0) ** 2, lambda z: 2 * np.maximum(z, 0)),
        "relu3": (lambda z: np.maximum(z, 0) ** 3, lambda z: 3 * np.maximum(z, 0) ** 2),
        "silu": (lambda z: z * sg(z), lambda z: sg(z) * (1 + z * (1 - sg(z)))),
        "gelu": (lambda z: z * Phi(z), lambda z: Phi(z) + z * np.exp(-z * z / 2) / math.sqrt(2 * math.pi)),
        "softplus": (lambda z: np.logaddexp(0, z), sg),
        "tanh": (np.tanh, lambda z: 1 - np.tanh(z) ** 2),
    }
    meas = {"softplus": 1.00, "relu3": 0.88, "relu2": 0.78, "relu1.5": 0.70, "silu": 0.59, "relu": 0.57, "gelu": 0.55, "tanh": -1.29}
    beta = 0.57                                  # downstream coupling, read off ReLU (a_A = 1, a_G = 0 exactly)
    def logs(f, r): return 0.5 * math.log((w * f(r * xi) ** 2).sum())
    print(f"  a_A = dlog RMS(phi)/dlog r, a_G = dlog RMS(phi')/dlog r, el_pred = (beta + a_G)/a_A with beta {beta}; "
          f"q0 = dlog(dA/dr)/dlog A (size growth exponent for a fixed-length step); q_M = q0 + (1 - n/m) el, n/m = 1/4")
    for name, (f, fp) in acts.items():
        out = []
        for r in [0.5, 1.0, 2.0]:
            e = 1e-2
            aA = (logs(f, r * math.exp(e)) - logs(f, r * math.exp(-e))) / (2 * e)
            aG = (logs(fp, r * math.exp(e)) - logs(fp, r * math.exp(-e))) / (2 * e)
            dAdr = lambda rr: (math.exp(logs(f, rr * math.exp(e))) - math.exp(logs(f, rr * math.exp(-e)))) / (rr * 2 * e)
            q0 = (math.log(abs(dAdr(r * math.exp(e)))) - math.log(abs(dAdr(r * math.exp(-e))))) / (2 * e) / aA if abs(aA) > 1e-9 else float("nan")
            elp = (beta + aG) / aA
            out.append(f"r={r}: a_A {aA:+.2f} a_G {aG:+.2f} el_pred {elp:+.2f} q0 {q0:+.2f} q_M {q0 + 0.75 * meas[name]:+.2f}")
        print(f"  {name:8s} (measured el {meas[name]:+.2f}) " + " | ".join(out))
    print("  size set by an offset instead of a scale: z = b + xi (r = 1); el_b = (dlog RMS(phi')/db) / (dlog RMS(phi)/db)")
    for name, (f, fp) in acts.items():
        out = []
        for b in [-1.0, 0.0, 1.0, 2.0]:
            e = 1e-2
            lA = lambda bb: 0.5 * math.log((w * f(bb + xi) ** 2).sum()); lG = lambda bb: 0.5 * math.log((w * fp(bb + xi) ** 2).sum())
            dA = (lA(b + e) - lA(b - e)) / (2 * e); dG = (lG(b + e) - lG(b - e)) / (2 * e)
            out.append(f"b={b:+.0f}: " + (f"{dG / dA:+.2f}" if abs(dA) > 1e-6 else "n/a (dA/db = 0)"))
        print(f"  {name:8s} " + "  ".join(out))


def c9():
    print("C9 persistence: momentum of a gradient = persistent per-row signal S (log-spread 0.7) + fresh iid noise each step;"
          " corr over rows of log h at step t and t+k, and the spread of log h")
    a, beta, T = 4, 0.95, 120
    for n in [32, 128, 512]:
        m = a * n
        for snr in [0.03, 0.1, 1.0]:             # per-entry signal size / per-entry noise size, per step
            r = np.random.default_rng(1)
            S = lognormal_rows(m, n, 0.7, r) * math.sqrt(n) * snr / math.exp(0.7 ** 2 / 2)
            u = np.zeros((m, n)); H = []
            for t in range(T):
                u = beta * u + S + r.standard_normal((m, n))
                if t in (99, 100, 104, 119): H.append(np.log(lev(u)))
            c1_, c5_, c20 = [np.corrcoef(H[0], H[j])[0, 1] for j in (1, 2, 3)]
            sig_inf = np.log(np.linalg.norm(S, axis=1)).std()
            hs = lev(u); hn = formula_h((u ** 2).sum(1), n)
            print(f"  n={n:3d} m={m:4d} snr {snr}: corr(log h) k=1 {c1_:.3f} k=5 {c5_:.3f} k=20 {c20:.3f}  "
                  f"row-length CV {cv(np.sqrt(hs)):.3f}  formula {cv(np.sqrt(hn)):.3f}  (signal-only sig {sig_inf:.2f})")
    print("  noise only (S = 0): corr(log h) k=20 for momentum 0.95")
    for n in [32, 128, 512]:
        m = a * n; r = np.random.default_rng(2); u = np.zeros((m, n)); H = []
        for t in range(T):
            u = beta * u + r.standard_normal((m, n))
            if t in (99, 119): H.append(np.log(lev(u)))
        print(f"    n={n}: corr {np.corrcoef(H[0], H[1])[0,1]:.3f}  row-length CV {cv(np.sqrt(np.exp(H[1]))):.3f}")


def c10():
    print("C10 toy size recurrence x_i <- x_i + eta f_i x_i^q, f lognormal CV 0.4 (mean 1); 1024 neurons, T steps;"
          " std(log x) at the end: persistent f vs f redrawn each step; closed form for q<1, persistent: sd(log f)/(1-q)")
    sdlf = math.sqrt(math.log(1 + 0.4 ** 2))
    for q in [-1.0, 0.0, 0.5, 0.9]:
        for T in [100, 1500]:
            eta = 0.02 * 1500 / T
            res = []
            for kind in ["persistent", "redrawn", "equal"]:
                r = np.random.default_rng(3); x = np.ones(1024)
                fp = np.exp(sdlf * r.standard_normal(1024) - sdlf ** 2 / 2)
                for t in range(T):
                    f = fp if kind == "persistent" else (np.exp(sdlf * r.standard_normal(1024) - sdlf ** 2 / 2) if kind == "redrawn" else 1.0)
                    x = x + eta * f * x ** q
                e = x ** 2; top = np.sort(e)[::-1][:102].sum() / e.sum()
                res.append(f"{kind} {np.log(x).std():.4f} (top10 {top:.3f})")
            lim = sdlf / (1 - q)
            print(f"  q={q:+.1f} T={T:4d} eta*T=30: " + "  ".join(res) + f"   persistent limit {lim:.4f}")
    print("  q = 1 (exponential growth): std(log x) persistent = eta*T*sd(f)-ish, redrawn ~ eta*sqrt(T)*sd(f)")
    for T in [100, 400, 1500]:
        eta = 2.0 / T; out = []
        for kind in ["persistent", "redrawn"]:
            r = np.random.default_rng(4); x = np.ones(1024); fp = np.exp(sdlf * r.standard_normal(1024) - sdlf ** 2 / 2)
            for t in range(T):
                f = fp if kind == "persistent" else np.exp(sdlf * r.standard_normal(1024) - sdlf ** 2 / 2)
                x = x * (1 + eta * f)
            out.append(f"{kind} {np.log(x).std():.4f}")
        print(f"    T={T} eta*T=2: " + "  ".join(out) + f"   predicted persistent ~{2*0.4:.3f}, redrawn ~{2*0.4/math.sqrt(T):.4f}")


def c11():
    print("C11 Muon raises the feedback exponent: toy x_i <- x_i + eta * l_i * x_i^q0 with step length l_i from msign of a")
    print("    gradient whose row i has size x_i^el (random directions); equal rows: l_i = const. Final std(log x), top10 share")
    m, n, T, eta = 512, 128, 300, 0.01
    for q0, el in [(0.5, 0.78), (0.0, 0.57), (-1.0, -1.29), (0.5, 0.0)]:
        out = []
        for kind in ["muon", "equal"]:
            r = np.random.default_rng(5); x = np.exp(0.1 * r.standard_normal(m))
            for t in range(T):
                G = lognormal_rows(m, n, 0.0, r) * (x ** el)[:, None]
                L = np.linalg.norm(polar(G), axis=1) if kind == "muon" else np.full(m, math.sqrt(n / m))
                x = x + eta * (L / math.sqrt(n / m)) * x ** q0
            e = x ** 2
            out.append(f"{kind}: std(log x) {np.log(x).std():.3f} top10 {np.sort(e)[::-1][:m//10].sum()/e.sum():.3f}")
        print(f"  q0 {q0:+.1f} el {el:+.2f} (q_M = q0 + 0.75 el = {q0 + 0.75*el:+.2f}): " + "  ".join(out))


def c12():
    print("C12 pointwise gate elasticity d log phi'(z) / d log phi(z) at single preactivations z (> 0): is the gate bigger where the output is bigger?")
    sg = lambda z: 1 / (1 + math.exp(-z)); Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    acts = {"relu": lambda z: max(z, 0), "relu2": lambda z: max(z, 0) ** 2, "relu3": lambda z: max(z, 0) ** 3,
            "silu": lambda z: z * sg(z), "gelu": lambda z: z * Phi(z), "softplus": lambda z: math.log1p(math.exp(z)), "tanh": math.tanh}
    zs = [0.25, 0.5, 1.0, 2.0, 3.0]
    for name, f in acts.items():
        d = lambda z, e=1e-4: (f(z + e) - f(z - e)) / (2 * e)
        out = []
        for z in zs:
            e = 1e-4
            el = (math.log(abs(d(z * math.exp(e)))) - math.log(abs(d(z * math.exp(-e))))) / (math.log(abs(f(z * math.exp(e)))) - math.log(abs(f(z * math.exp(-e)))))
            out.append(f"z={z}: {el:+.3f}")
        print(f"  {name:8s} " + "  ".join(out))


def c13():
    print("C13 downstream coupling. (a) Euler identity for ReLU^p: w1_i.g1_i = p * (w2_i.g2_i) for every hidden neuron i, any data, any loss")
    r = np.random.default_rng(13)
    X = r.standard_normal((200, 64)); Y = r.standard_normal((200, 8)); W1 = r.standard_normal((32, 64)) / 8; W2 = r.standard_normal((8, 32)) / 6
    sg = lambda z: 1 / (1 + np.exp(-z))
    acts = {"relu": (1, lambda z: np.maximum(z, 0), lambda z: (z > 0) * 1.0),
            "relu1.5": (1.5, lambda z: np.maximum(z, 0) ** 1.5, lambda z: 1.5 * np.maximum(z, 0) ** 0.5),
            "relu2": (2, lambda z: np.maximum(z, 0) ** 2, lambda z: 2 * np.maximum(z, 0)),
            "relu3": (3, lambda z: np.maximum(z, 0) ** 3, lambda z: 3 * np.maximum(z, 0) ** 2),
            "silu (p=1 tried)": (1, lambda z: z * sg(z), lambda z: sg(z) * (1 + z * (1 - sg(z))))}
    for name, (pw, f, fp) in acts.items():
        H = X @ W1.T; A = f(H); D = A @ W2.T - Y                      # loss 0.5 |W2 phi(W1 x) - y|^2 summed over samples
        g2 = D.T @ A; g1 = ((D @ W2) * fp(H)).T @ X
        lhs = (W1 * g1).sum(1); rhs = pw * (W2 * g2).sum(0)
        print(f"  {name:16s} max |w1.g1 - p w2.g2| / max|w1.g1| = {np.abs(lhs - rhs).max() / np.abs(lhs).max():.2e}")
    print("  (b) paired snapshots (seed 0, same settings, step 750): slope of log|l1 momentum row i| on log|l2 momentum column i|")
    import torch
    for act in ["relu2", "gelu", "swiglu"]:
        a = torch.load(os.path.join(HERE, "snaps", f"mlpx_{act}.pt"), weights_only=False)["l1.weight"].double().numpy()
        b = torch.load(os.path.join(HERE, "snaps", f"wide_mlpx_{act}.pt"), weights_only=False)["l2.weight"].double().numpy()
        y2 = np.log(np.linalg.norm(b, axis=0)); k = b.shape[1]
        parts = [("l1", a)] if a.shape[0] == k else [("l1 up", a[:k]), ("l1 gate", a[k:])]
        for nm, P in parts:
            y1 = np.log(np.linalg.norm(P, axis=1)); sl = np.polyfit(y2, y1, 1)[0]
            rk = lambda v: np.argsort(np.argsort(v)); sp = np.corrcoef(rk(y1), rk(y2))[0, 1]
            print(f"    {act:7s} {nm:8s} slope {sl:.3f}  Pearson {np.corrcoef(y1, y2)[0,1]:.3f}  Spearman {sp:.3f}  sd log row {y1.std():.2f}  sd log col {y2.std():.2f}")


def _moe(case):                                     # one full-shape matrix per run (~18 s each: eigh of 3072x3072 in float64)
    import time
    r = np.random.default_rng(14); t0 = time.time(); dff, d = 3072, 7168
    hid = np.exp(0.7 * r.standard_normal(dff))       # hidden-neuron gradient sizes, sig 0.7 (SwiGLU-like, c6)
    if case == "up":                                 # gate and up have the same wide shape d_ff x d_model: rows = hidden neurons
        M = hid[:, None] * r.standard_normal((dff, d))
        e, V = np.linalg.eigh(M @ M.T); O = (V * e ** -0.5) @ V.T @ M
        rw = np.linalg.norm(O, axis=1); cl = np.linalg.norm(O, axis=0)
        print(f"  gate/up {dff}x{d} (wide): hidden-neuron rows min {rw.min():.12f} max {rw.max():.12f}   "
              f"residual columns CV {cv(cl):.3f}, sum of squares {(cl**2).sum():.3f} (= {dff})  ({time.time()-t0:.1f} s)")
        return
    rs = np.ones(d)
    if case == "sig": rs = np.exp(0.5 * r.standard_normal(d))
    if case == "massive": rs[:16] = 10.0
    M = rs[:, None] * r.standard_normal((d, dff)) * hid[None, :]        # down: d_model x d_ff (tall)
    e, V = np.linalg.eigh(M.T @ M); O = M @ ((V * e ** -0.5) @ V.T)
    rw = np.linalg.norm(O, axis=1); cl = np.linalg.norm(O, axis=0)
    extra = f"  massive dims' step {rw[:16].mean():.3f} vs others {rw[16:].mean():.3f} (mean sqrt(n/m) {math.sqrt(dff/d):.3f})" if case == "massive" else ""
    print(f"  down {d}x{dff} (tall), residual sizes {case}: hidden-neuron columns min {cl.min():.12f} max {cl.max():.12f}   residual rows CV {cv(rw):.3f}"
          f"{' (formula (1-n/m)*0.5 = %.3f)' % ((1 - dff/d) * 0.5) if case == 'sig' else ''}{extra}  ({time.time()-t0:.1f} s)")


def c14():
    print("C14 fine-grained MoE expert, full shapes d_ff 3072, d_model 7168, exact msign in float64. Run c14 c14b c14c c14d one at a time.")
    _moe("up")
    print("  rank caveat (1/16 scale, 192x448 wide): rows are exactly equal only if the momentum has rank = d_ff (enough tokens per expert)")
    r = np.random.default_rng(141)
    for k in [64, 128, 191, 192, 1000]:
        M = np.exp(0.7 * r.standard_normal(192))[:, None] * (r.standard_normal((192, k)) @ r.standard_normal((k, 448)))
        U, S, Vt = np.linalg.svd(M, full_matrices=False); kk = (S > S[0] * 1e-10).sum(); O = U[:, :kk] @ Vt[:kk]
        rw = np.linalg.norm(O, axis=1); print(f"    momentum rank {min(k,192):4d}: hidden-row step CV {cv(rw):.4f} (min {rw.min():.3f})")
def c14b(): _moe("equal")
def c14c(): _moe("sig")
def c14d(): _moe("massive")


def ns5(G, steps=5):                               # Keller Jordan's quintic Newton-Schulz (Muon default), float32
    a, b, c = 3.4445, -4.7750, 2.0315
    X = (G / (np.linalg.norm(G) + 1e-7)).astype(np.float32); tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


def c15():
    print("C15 practical Newton-Schulz (NS5): row^2 = sum_k U_ik^2 f(sig_k)^2 on a wide matrix, so every row length lies in [min f, max f]")
    import torch, time
    r = np.random.default_rng(15)
    for tag, k in [("wide_mlpx_relu2", "l2.weight"), ("wide_mlpx_swiglu", "l2.weight"), ("wide_lm_swiglu", None), ("mlpx_relu2", "l1.weight"), ("mlpx_swiglu", "l1.weight")]:
        d = torch.load(os.path.join(HERE, "snaps", tag + ".pt"), weights_only=False); k = k or max(d, key=lambda q: d[q].numel())
        M = d[k].double().numpy(); O = polar(M); N = ns5(M).astype(np.float64)
        f = np.linalg.svd(N, compute_uv=False); side = 1 if M.shape[0] > M.shape[1] else 0   # hidden side: rows if tall, columns if wide
        hid = lambda Z: np.linalg.norm(Z, axis=1 if M.shape[0] > M.shape[1] else 0); oth = lambda Z: np.linalg.norm(Z, axis=0 if M.shape[0] > M.shape[1] else 1)
        print(f"  {tag} {k} {M.shape}: short side (exact 1): NS5 lengths [{oth(N).min():.3f}, {oth(N).max():.3f}] CV {cv(oth(N)):.4f}, f in [{f.min():.3f}, {f.max():.3f}]"
              f" | long side CV exact {cv(hid(O)):.3f} NS5 {cv(hid(N)):.3f}  corr {np.corrcoef(hid(O), hid(N))[0,1]:.3f}")
    t0 = time.time(); dff, dm = 3072, 7168
    M = (np.exp(0.7 * r.standard_normal(dff))[:, None] * r.standard_normal((dff, dm))).astype(np.float32); N = ns5(M)
    rw = np.linalg.norm(N.astype(np.float64), axis=1)
    print(f"  MoE expert up {dff}x{dm}, hidden sizes sig 0.7: NS5 hidden-row lengths [{rw.min():.3f}, {rw.max():.3f}] CV {cv(rw):.4f}; "
          f"corr with log row size {np.corrcoef(rw, np.log(np.linalg.norm(M.astype(np.float64), axis=1)))[0,1]:+.3f}  ({time.time()-t0:.1f} s)")
    sz = np.linalg.norm(M.astype(np.float64), axis=1); q = np.quantile(sz, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, sz, side="right") - 1, 0, 9)
    print("    NS5 step length by row-size decile (smallest -> largest): " + " ".join(f"{rw[b == i].mean():.3f}" for i in range(10)))
    x = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0])     # the scalar map f(sigma) after 5 NS5 steps (sigma = singular value / |M|_F)
    for _ in range(5): x = 3.4445 * x - 4.7750 * x ** 3 + 2.0315 * x ** 5
    print("    f(sigma) at sigma = 1e-3 2e-3 5e-3 1e-2 2e-2 5e-2 0.1 0.2 0.5 1: " + " ".join(f"{v:.3f}" for v in x))


def c16():
    print("C16 dense SwiGLU up/gate (tall): predicted row-length CV from the size-spread formula, at full shapes (formula needs only row sizes)")
    r = np.random.default_rng(16)
    shapes = [("DeepSeek-V3 dense 18432x7168", 18432, 7168), ("Llama-3 8B 14336x4096", 14336, 4096),
              ("fused [gate;up] 36864x7168", 36864, 7168), ("fused [gate;up] 28672x4096", 28672, 4096), ("mlpx 1024x256 (a=4)", 1024, 256)]
    for name, m, n in shapes:
        out = []
        for sig in [0.3, 0.5, 0.6, 0.7, 0.8]:
            s2 = np.exp(2 * sig * r.standard_normal(m)); h = formula_h(s2, n); out.append(f"sig {sig}: {cv(np.sqrt(h)):.3f}")
        print(f"  {name} (a={m/n:.2f}, 1-n/m={1-n/m:.3f}): " + "  ".join(out))
    print("  exact SVD check at 1/16 scale (1152x448, 896x256), sig 0.7: ", end="")
    for m, n in [(1152, 448), (896, 256)]:
        M = lognormal_rows(m, n, 0.7, r); L = np.sqrt(lev(M)); hf = formula_h(np.linalg.norm(M, axis=1) ** 2, n)
        print(f"{m}x{n} exact {cv(L):.3f} formula {cv(np.sqrt(hf)):.3f}   ", end="")
    print()
    print("  SwiGLU feedback: phi = silu(r xi1) * (r xi2), both rows scaled by r; q0 = dlog(dA/dr)/dlog A by quadrature")
    xi = np.linspace(-10, 10, 40001); w = np.exp(-xi ** 2 / 2); w = w / w.sum()
    sg = lambda z: 1 / (1 + np.exp(-z))
    def logA(rr): return 0.5 * math.log((w * (rr * xi * sg(rr * xi)) ** 2).sum()) + math.log(rr)   # RMS of silu(r xi1) * RMS of r xi2
    e = 1e-2
    for rr in [0.5, 1.0, 2.0]:
        aA = (logA(rr * math.exp(e)) - logA(rr * math.exp(-e))) / (2 * e)
        dAdr = lambda q: (math.exp(logA(q * math.exp(e))) - math.exp(logA(q * math.exp(-e)))) / (q * 2 * e)
        q0 = (math.log(dAdr(rr * math.exp(e))) - math.log(dAdr(rr * math.exp(-e)))) / (2 * e) / aA
        out = [f"r={rr}: a_A {aA:.2f} q0 {q0:.3f}"]
        for el in [0.78, 0.86]:
            thr = 1 / (1 - (1 - q0) / el) if el > 1 - q0 else float("inf")
            out.append(f"el {el}: q_M at a=2.57 {q0 + (1-1/2.571)*el:.2f}, a=3.5 {q0 + (1-1/3.5)*el:.2f}, a=7 {q0 + (1-1/7)*el:.2f}, runaway aspect > {thr:.2f}")
        print("    " + " | ".join(out))


if __name__ == "__main__":
    for s in sys.argv[1:]: globals()[s]()
