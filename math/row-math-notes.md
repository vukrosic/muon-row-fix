# Row math: why Muon's rows are uneven on tall matrices, and when it hurts

2026-10-19. Math only: identities, derivations, bounds, and tiny exact numerical checks. No training runs.
Checks: `row_math_checks.py` (numpy float64, CPU). Run one section with `taskpolicy -b nice -n 20 python3 row_math_checks.py c1` (each under 25 s).
Training facts cited as "§N" (§8, §9–22) come from small training runs (a 1024×256 MLP layer, 1500 steps, up to 8 seeds) that are not in this repo; they are quoted, not re-measured here.

Labels: **PROVEN** = derivation given here. **CHECKED** = numbers from the named check section. **CONJECTURE** = not proven or checked.

Notation. A weight W has shape m×n (rows = output neurons, as in PyTorch `Linear.weight`). Tall means m > n, aspect a = m/n.
M = Muon's momentum for W. A = MᵀM. O = msign(M) = UVᵀ (Muon's step before lr). Row i of M is m_i = s_i z_i with s_i = |m_i|, |z_i| = 1.
h_i = m_iᵀA⁻¹m_i (leverage of row i). ℓ_i = |row i of O| (Muon's step length for neuron i). R = roweq(O) = Aurora / U-NorMuon equal-rows step.

## Summary: findings that hold at every model size

1. **Budget.** On a tall matrix, Muon's squared step lengths per neuron are the leverages h_i. They lie in [0, 1] and sum to exactly n, so the average is n/m. A neuron's step grows only by taking step from others. (PROVEN)
2. **Exact sensitivity.** If one neuron's momentum row is scaled up by 1%, its Muon step grows by (1 − h_i)%. On average that is (1 − n/m)%: 75% of any gradient-size difference passes into the step at aspect 4, 63% at aspect 2.7. Equal rows pass 0%. (PROVEN, exact)
3. **What sets row length.** Two things: the neuron's gradient size, and how rare its gradient direction is (energy along weak singular directions counts more). If row directions are generic, only the aspect and σ (spread of log gradient-row sizes) enter; width does not. Small σ: CV ≈ (1 − n/m)·σ. At σ = 0.6 (measured for SwiGLU) the formula predicts CV 0.33 for DeepSeek-V3's dense up 18432×7168 and 0.39 for Llama-3 8B's 14336×4096; fusing gate and up adds about 0.1. (PROVEN in the large-width limit, CHECKED to ±0.004; σ at full size is CONJECTURE)
4. **Noise alone gives nearly equal rows at scale.** For a pure-noise momentum, row-length CV = √((1 − n/m)/(2n)): 0.038 at n = 256, 0.027 at n = 512. So in large models almost all of Muon's row unevenness comes from persistent per-neuron gradient-size differences, and it repeats step after step. (PROVEN, CHECKED)
5. **Which activations feed back.** For ReLU^p the gate is an exact power of the output, φ' = p·φ^((p−1)/p), so bigger neurons get bigger gradients, for any input distribution. tanh: φ' = 1 − φ² (a big neuron gets a smaller gradient). GLUs are degree 2 in their two rows, like ReLU². ReLU^p also obeys an exact balance identity, w1_i·g1_i = p·(w2_i·g2_i), so gradient descent ties each neuron's output weights to its input weights (β = 1). Under Muon the snapshots give β ≈ 0.57–0.69. (PROVEN; β under Muon CHECKED by proxy)
6. **Persistent unevenness compounds; redrawn unevenness averages out.** In a growth model x ← x + η f x^q, a fixed per-neuron factor f leaves a permanent spread sd(log f)/(1 − q). A factor redrawn every step leaves sd(f)/((1 − q)√T), which is 39× smaller at T = 1500. Self-amplification (q → 1) magnifies the persistent case without limit. (PROVEN for the model, CHECKED)
7. **Muon adds feedback; equal rows remove it.** In that growth model, Muon's per-neuron steps raise the feedback exponent from q₀ (the activation's own) to q₀ + (1 − n/m)·el, where el = the gradient-vs-size exponent. ReLU²: 0.50 → 1.08, past the runaway line q = 1 (from aspect 2.8). SwiGLU: q₀ ≈ 0.54, runaway from aspect 2.1–2.8; Llama-3 8B (3.5) and all fused layouts are past it, DeepSeek-V3 dense (2.57) sits on it. tanh: el < 0, so Muon evens neurons out more than equal rows do. (PROVEN for the toy model, CHECKED. The mapping to real networks is a CONJECTURE.)
8. **Equal rows cost little direction; fine-grained MoE needs no fix.** The cosine between Muon's step and the equal-rows step is exactly 1/√(1 + CV²): CV 0.4 gives 0.93 (22°). On wide matrices rows are exactly equal. At DeepSeek expert shapes (gate/up 3072×7168, down 7168×3072) every hidden neuron gets a step of exactly 1.000000000000 on all three matrices, given full-rank momentum. The uneven rows of down are residual-stream dimensions; no activation follows them, so the §2 gate loop cannot act on them (PROVEN by architecture), and RMSNorm's projection makes their feedback zero or negative (CONJECTURE). Practical 5-step Newton–Schulz keeps short-side lengths within [min f, max f] and adds back CV ≈ 0.1 at full MoE shape, but not monotone in neuron size. (PROVEN, CHECKED)

---

## 1. Exact row lengths (question 1)

**Plain version.** On a tall matrix Muon shares a fixed budget of step among the neurons. A neuron gets more if its gradient is bigger or points in a direction few other neurons use.
Example: M = [[2,0],[1,0],[0,1],[0,1]] (4 neurons, 2 inputs). MᵀM = diag(5, 2), so the leverages are 4/5, 1/5, 1/2, 1/2, summing to n = 2. Muon's step lengths are 0.89, 0.45, 0.71, 0.71. Neuron 1 has twice neuron 2's gradient and gets twice the step. Neurons 3 and 4 have the same gradient size as neuron 2 but get 0.71 instead of 0.45: their direction carries less total energy (2 vs 5).

### 1.1 The identity (PROVEN)

For M with full column rank, the polar factor is O = M A^(−1/2) (because M = O·A^(1/2) and OᵀO = I). So row i of O is the vector A^(−1/2) m_i, and

    ℓ_i² = m_iᵀ A⁻¹ m_i = h_i.

OOᵀ = M A⁻¹ Mᵀ = P, the projector onto M's column space. So h_i = P_ii ∈ [0, 1], Σ h_i = tr P = rank = n, mean n/m.
The same identity says Muon's step for neuron i is its gradient row in A-whitened coordinates: o_i = A^(−1/2) m_i.

CHECKED (c1): 512×128 and 2048×512 with lognormal row sizes: max |ℓ_i² − h_i| = 2.8e−15 and 5.1e−15; Σh = 128.000000 and 512.000000; |OᵀO − I| ≤ 4.4e−15.

### 1.2 What makes h_i big (PROVEN)

Three exact forms:

- **Spectral.** With A = Σ_k σ_k² v_k v_kᵀ: h_i = Σ_k (m_i·v_k)²/σ_k². A row's energy along a weak singular direction (small σ_k) counts more than the same energy along a strong one.
- **Leave-one-out (Sherman–Morrison).** A = A₋ᵢ + m_i m_iᵀ gives h_i = q_i/(1 + q_i), with q_i = s_i² · z_iᵀA₋ᵢ⁻¹z_i. Size enters as s_i². Rarity enters as z_iᵀA₋ᵢ⁻¹z_i: large when the other rows have little energy in z_i's direction. h_i → 1 as s_i → ∞.
- **Sandwich.** s_i²/σ_max² ≤ h_i ≤ s_i²/σ_min². So for any two rows, (s_i/s_j)/κ ≤ ℓ_i/ℓ_j ≤ (s_i/s_j)·κ, with κ = σ_max/σ_min the condition number of M. A well-conditioned momentum keeps row-length ratios close to gradient-size ratios.

CHECKED (c1): Sherman–Morrison matches h_i to 1.1e−16. CHECKED (c5), 1024×256, all rows of equal size:

| half the rows share direction b with weight w | mean h, sharing rows | mean h, other rows |
|---|---|---|
| w = 0 | 0.2494 | 0.2506 |
| w = 0.5 | 0.2240 | 0.2760 |
| w = 0.8 | 0.1591 | 0.3409 |

Rule: the shared component counts as almost zero. With effective size² = 1 − w² = 0.36 for the sharing rows, the formula of §1.5 predicts 0.158 / 0.342 (worked by hand from Σh = n). One row of normal size pointing into an otherwise empty direction gets h = 0.970, against a median of 0.249.

### 1.3 Exact sensitivity of a row's step to its gradient (PROVEN)

Scale row i of M by t, with everything else fixed. A₋ᵢ does not change, so q_i(t) = t² q_i exactly, and

    d log h_i / d log t = 2(1 − h_i),     d log ℓ_i / d log t = 1 − h_i.

Since Σh = n is fixed, the other rows lose exactly d h_i = 2h_i(1 − h_i) d log t in total.
Meaning: Muon passes a fraction (1 − h_i) of any relative gradient-size change of a neuron into its step. Plain momentum SGD passes 1. Equal rows pass 0. On average over rows that fraction is 1 − n/m.
Example at aspect 4, h = 0.25: a 10% bigger gradient gives a 7.5% bigger step. With the §1.5 formula, doubling the gradient multiplies the step by 1.51.

CHECKED (c2): at rows with h = 0.0003, 0.1355 and 0.9903, the finite-difference elasticity is 0.99971, 0.86445 and 0.00970. 1 − h is 0.99971, 0.86445 and 0.00970. The other rows lose −0.23435 vs predicted −0.23435.

### 1.4 Pure noise: leverage is Beta-distributed, unevenness vanishes with width (PROVEN)

If M has i.i.d. Gaussian entries, its column space is a uniformly random n-dimensional subspace of R^m. Then h_i = |P e_i|² is the squared projection of a fixed unit vector onto a random subspace, so

    h_i ~ Beta(n/2, (m − n)/2),   mean n/m,   CV(h) = √(2(m − n)/(n(m + 2))),   CV(ℓ) ≈ √((1 − n/m)/(2n)).

CHECKED (c3):

| shape | CV(h) measured / exact | CV(ℓ) measured / approx |
|---|---|---|
| 256×64 | 0.1505 / 0.1525 | 0.0757 / 0.0765 |
| 1024×256 | 0.0754 / 0.0765 | 0.0378 / 0.0383 |
| 2048×512 | 0.0543 / 0.0541 | 0.0272 / 0.0271 |
| 2048×256 (a = 8) | 0.0832 / 0.0826 | 0.0416 / 0.0413 |

Consequence: the measured row CVs of 0.13–0.86 (c6, real momenta) are far above the noise level of 0.04. Almost all of them come from structure (gradient-size spread), not from sampling noise. At larger width, the noise part shrinks further.

### 1.5 The size-spread formula and the aspect rule

Formula (also in §8): h_i ≈ c s_i²/(1 + c s_i²), where c solves Σ_i c s_i²/(1 + c s_i²) = n. The solution is unique, because the left side rises from 0 to m > n as c goes from 0 to ∞.

Derivation (PROVEN in the large-n limit, for row directions z_i that are random and independent of the sizes): z_iᵀA₋ᵢ⁻¹z_i concentrates at tr(A₋ᵢ⁻¹)/n with relative error O(1/√n) (standard quadratic-form concentration). tr(A₋ᵢ⁻¹) differs from tr(A⁻¹) by one rank-one term, a relative O(1/n). So every row has the same c up to O(1/√n), and Σh = n fixes it.

Properties (PROVEN from the formula):
- Scale-free. s → k·s gives c → c/k². Only the distribution of log s and the aspect a matter.
- Small spread. log ℓ has slope 1 − h ≈ 1 − n/m in log s (as in §1.3), so **CV(ℓ) ≈ (1 − n/m)·σ**, with σ = std(log s).
- a → 1: h → 1 for every row, so a square matrix gives exactly equal rows.
- a → ∞: mean h = n/m → 0, so c·s_i² → 0 and ℓ_i ∝ s_i. Very tall Muon keeps the gradient's row-size ratios, like plain normalized momentum.
- Large spread. Big rows cap at ℓ = 1 and small rows stay ∝ s, so the CV saturates below (1 − n/m)·σ.

CHECKED (c4), n = 256, lognormal row sizes, row-length CV as exact SVD / formula / small-spread rule:

| aspect | σ = 0.25 | σ = 0.5 | σ = 1.0 |
|---|---|---|---|
| 1.5 | 0.086 / 0.082 / 0.083 | 0.155 / 0.151 / 0.167 | 0.289 / 0.288 / 0.333 |
| 2 | 0.121 / 0.118 / 0.125 | 0.240 / 0.241 / 0.250 | 0.410 / 0.408 / 0.500 |
| 2.7 | 0.154 / 0.153 / 0.157 | 0.294 / 0.294 / 0.315 | 0.480 / 0.480 / 0.630 |
| 4 | 0.187 / 0.185 / 0.188 | 0.357 / 0.355 / 0.375 | 0.611 / 0.609 / 0.750 |
| 8 | 0.214 / 0.213 / 0.219 | 0.426 / 0.426 / 0.438 | 0.776 / 0.775 / 0.875 |

The formula is within 0.004 of exact everywhere. The rule (1 − n/m)·σ is good to σ ≈ 0.5 and overshoots above that, because of the cap at 1.

---

## 2. The feedback loop: gradient size vs neuron size (question 2)

**Plain version.** For ReLU², a neuron that fires twice as strongly also gets a gradient that is √2 times bigger, so Muon gives it a bigger step. For tanh, a neuron that is more active gets a smaller gradient.
Example: in ReLU², doubling a neuron's input weights makes its output 4× and its gate 2×, so gradient ∝ output^(1/2). In tanh, a neuron with RMS output 0.6 has mean gate exactly 1 − 0.36 = 0.64; at RMS 0.8 the mean gate is 0.36.

### 2.1 The gate as a function of the output (PROVEN, any input distribution)

The gradient of a neuron's input-weight row is g_i = E[δ_i · φ'(z_i) · x], where δ_i is the error that reaches neuron i from above. The gate φ' is what the activation contributes. For several activations, φ' is an exact function of φ:

| activation | exact identity | gate elasticity d log φ'/d log φ | sign |
|---|---|---|---|
| ReLU^p (z > 0) | φ' = p·φ^((p−1)/p) | (p − 1)/p: ReLU 0, ReLU^1.5 1/3, ReLU² 1/2, ReLU³ 2/3 | self-amplifying (p > 1) |
| Softplus | φ' = 1 − e^(−φ) | φ/(e^φ − 1) ∈ (0, 1): 0.69 at z = 0, 0.48 at z = 1, 0.15 at z = 3 | self-amplifying |
| tanh | φ' = 1 − φ² | −2φ²/(1 − φ²): −0.13 at z = 0.25, −2.76 at z = 1 | self-limiting |
| ReGLU, h = ReLU(w_g·x)·(w_u·x) | degree 2 in (w_g, w_u) jointly | 1/2 when both rows grow together, as ReLU² | self-amplifying |

Proof for ReLU^p: on z > 0, φ = z^p and φ' = p z^(p−1) = p φ^((p−1)/p). Softplus: e^φ = 1 + e^z, so σ(z) = 1 − e^(−φ). tanh: tanh' = 1 − tanh². ReGLU: scaling both rows by t scales h by t² and each row's gradient by t (the up row's gradient carries the gate's output, and the gate row's gradient carries w_u·x). SwiGLU equals ReGLU for large gate inputs. That is a structural reason why SwiGLU behaves like ReLU² in §9 (CONJECTURE for SwiGLU's exact exponent).

A clean corollary for tanh (PROVEN): E[φ'] = 1 − E[φ²] = 1 − A², where A = the neuron's RMS output. So the mean gate falls exactly as the output rises, with elasticity −2A²/(1 − A²). The measured slope el = −1.29 (§20) fits a typical A ≈ 0.62 (CONJECTURE: A was not measured).

SiLU and GELU have no such identity (φ is not monotone). CHECKED (c12), pointwise elasticity at z = 0.25 / 0.5 / 1 / 2 / 3:
SiLU +0.18 / +0.25 / +0.26 / +0.07 / −0.08; GELU +0.23 / +0.28 / +0.17 / −0.18 / −0.09. Both are weakly self-amplifying near zero and neutral or slightly self-limiting for large inputs. GELU turns negative sooner, which fits SiLU's measured el 0.59 being above GELU's 0.55.

### 2.2 From the gate to the measured slope el (CHECKED against §20)

§20 measured el = log-log slope of a neuron's l1 gradient-row norm against its RMS activation. Model: a neuron's preactivation scale r (its input-weight size) sets the activation RMS ∝ r^(a_A) and the gate RMS ∝ r^(a_G). The error reaching it from above scales as r^β (big neurons also get big downstream weights). Then

    el = (β + a_G)/a_A.

For ReLU^p, a_A = p and a_G = p − 1 exactly (homogeneity), so el = (β + p − 1)/p. ReLU (a_A = 1, a_G = 0) gives β = 0.57 directly. That one number then predicts the other three ReLU^p values with no further fitting (c8):

| | ReLU^1.5 | ReLU² | ReLU³ |
|---|---|---|---|
| predicted (0.57 + p − 1)/p | 0.71 | 0.79 | 0.86 |
| measured el (§20) | 0.70 | 0.78 | 0.88 |

For non-homogeneous activations, a_A and a_G depend on the input scale r (c8, Gaussian inputs): SiLU 0.60–0.64 (measured 0.59), GELU 0.57–0.65 (measured 0.55), Softplus 0.85 at r = 2 and 1.37 at r = 1 (measured 1.00).
The scale picture fails for tanh: it predicts +0.40 to +0.46, but tanh measures −1.29. If instead a tanh neuron is big because it is offset into saturation (z = b + ξ), the slope is −1.58 at |b| = 1 and −5.75 at b = 2 (c8). So tanh neurons are big by saturating, which shrinks their gate (§2.1 identity). β is the downstream weight coupling. Section 6 below derives β = 1 under plain gradient descent and reads a proxy from the snapshots (β ≈ 0.69 for ReLU²).

### 2.3 A growth model with Muon in it (PROVEN for the model; mapping to real nets is CONJECTURE)

Let x_i be neuron i's size (RMS activation). Each step moves its weights by a step of length ℓ_i in a direction that grows it, so x_i ← x_i + η ℓ_i x_i^(q₀). Here q₀ is the activation's own growth exponent for a fixed-length step: q₀ = d log(dA/dr)/d log A, which is 1 − 1/p for ReLU^p (c8: ReLU 0, ReLU² 0.50, ReLU³ 0.67; tanh −2.9 at r = 1). Then:

- **Equal rows:** ℓ_i is the same for all i, so the exponent is q₀.
- **Muon:** the gradient row size is ∝ x_i^el, and by §1.3 ℓ_i ∝ (gradient size)^(1 − h), so ℓ_i ∝ x_i^((1 − n/m)·el). The exponent becomes **q_M = q₀ + (1 − n/m)·el**.

Linear stability (PROVEN): write log x_i = mean + ε_i. Then dε_i/dt = η·x̄^(q−1)·(q − 1)·ε_i. Differences between neurons shrink when q < 1 and grow when q > 1. So q = 1 is the runaway line, where the biggest neurons pull away.

At aspect 4 (mlpx, measured el):

| activation | q₀ (equal rows) | q_M (Muon) | sensitivity to persistent unevenness, equal → Muon: 1/(1−q) | measured gain from equal rows (§9, §13, §21, §22) |
|---|---|---|---|---|
| ReLU³ | 0.67 | 1.33 | 3.0 → runaway | −0.022 |
| ReLU² | 0.50 | 1.08 | 2.0 → runaway | −0.012 (8 seeds) |
| ReLU^1.5 | 0.33 | 0.86 | 1.5 → 7.1 | −0.0075 |
| SiLU (r = 1) | 0.15 | 0.59 | 1.2 → 2.4 | −0.0047 (8 seeds) |
| ReLU | 0 | 0.43 | 1.0 → 1.75 | −0.001 (2–3 seeds) |
| GELU (r = 1) | 0.01 | 0.42 | 1.0 → 1.7 | +0.005 (2–3 seeds) |
| tanh (r = 1) | −2.9 | −3.9 | 0.26 → 0.20 | −0.001 |

The ordering by q_M matches the order of gains for the ReLU^p family and SiLU. GELU and ReLU are predicted to sit near SiLU, which fits §22's open point that their "no gain" rests on 2–3 seeds. For ReLU², Muon crosses the runaway line when (1 − n/m)·0.78 > 0.5, that is at aspect > 2.8 (CONJECTURE: toy threshold). SwiGLU up/gate sit at aspect 2.69 (Llama-2 7B), 2.57 (DeepSeek-V3 dense) and 3.5 (Llama-3 8B); section 8 below works out SwiGLU's own threshold.

CHECKED (c11): a direct simulation of the toy with Muon in the loop. 512×128, 300 steps; each step uses msign of a gradient with random row directions and row sizes x_i^el; neuron log-sizes start with spread 0.10.

| case | Muon: final std(log x), top-10% share | equal rows |
|---|---|---|
| ReLU²-like (q₀ 0.5, el 0.78) | 0.112, 0.143 (spread grew) | 0.038, 0.114 (spread shrank) |
| ReLU-like (q₀ 0, el 0.57) | 0.043, 0.116 | 0.024, 0.109 |
| tanh-like (q₀ −1, el −1.29) | 0.005, 0.102 | 0.014, 0.105 |
| control (el = 0) | 0.038, 0.114 | 0.038, 0.114 |

Muon only differs from equal rows when gradient size depends on neuron size (control row identical). The direction of the difference follows the sign of el. Both match §18 (ReLU² top-10% share 0.34 vs 0.29; tanh no difference).

---

## 3. Persistence: why fixed unevenness compounds and redrawn unevenness averages out (question 3)

**Plain version.** A neuron that always gets 20% more step moves 20% further in total. A neuron whose bonus is redrawn every step ends up within about 1% of the others after 1500 steps.

### 3.1 Compounding (PROVEN for the growth model, CHECKED)

Take x_i ← x_i + η f_i x_i^q with a per-neuron multiplier f_i (mean 1). In continuous time, x^(1−q) is linear in the summed multiplier:

    x_i(T)^(1−q) = x_i(0)^(1−q) + (1 − q)·η·Σ_t f_i(t).

- **Persistent** f_i: Σ_t f_i = T f_i. For q < 1 and large ηT, x_i ∝ f_i^(1/(1−q)), so **std(log x) → sd(log f)/(1 − q)**. This spread is permanent and does not shrink with time. For q = 1, log x_i = ηT f_i, so the spread grows linearly in T. For q > 1, the neuron with the largest f blows up first.
- **Redrawn** f_i(t), i.i.d.: Σ_t f_i = T ± sd(f)·√T, so **std(log x) ≈ sd(f)/((1 − q)√T)**, which goes to 0.
- Ratio of the two: √T·sd(log f)/sd(f) ≈ √T. That is 39× at T = 1500. Self-amplification (q near 1) multiplies both by 1/(1 − q), so it turns a small persistent bias into a large spread. Negative q shrinks it.

CHECKED (c10), 1024 neurons, f lognormal with CV 0.4, ηT = 30:

| q | persistent, T = 1500 | redrawn, T = 100 | redrawn, T = 1500 | persistent limit sd(log f)/(1−q) |
|---|---|---|---|---|
| −1 | 0.190 | 0.020 | 0.005 | 0.193 |
| 0 | 0.373 | 0.039 | 0.010 | 0.385 |
| 0.5 | 0.718 (top 10% hold 0.57) | 0.074 | 0.020 | 0.771 |
| 0.9 | 2.82 (top 10% hold 1.00) | 0.280 | 0.078 | 3.85 (not yet reached) |

At q = 1, ηT = 2: persistent 0.770 / 0.784 / 0.787 at T = 100 / 400 / 1500 (predicted ~0.80). Redrawn 0.077 / 0.040 / 0.020 (predicted 0.080 / 0.040 / 0.021).
This is the §21 result in miniature: fixed random unevenness (inj) costs as much as Muon, and moving unevenness (injr) costs about nothing. tanh (very negative q) pays little even for fixed unevenness.

### 3.2 Muon's unevenness is the persistent kind, more so at larger width (PROVEN under a signal + i.i.d. noise model, CHECKED)

Model: each step's gradient = persistent per-row signal S (row sizes set by neuron sizes, §2) + fresh i.i.d. noise of variance ν per entry. Momentum β gives u = S/(1 − β) + noise with variance ν/(1 − β²) per entry. Then

    s_i² = |S_i|²/(1 − β)² + n·ν/(1 − β²)·(1 ± √(2/n)).

Both terms scale with n, so the deterministic spread of log s_i does not depend on width. The fluctuating part has relative size √(2/n) and averages away as width grows. So (a) the per-neuron step sizes Muon assigns become more persistent as width grows, and (b) the noise floor n·ν/(1 − β²) compresses the spread when the signal is weak.

CHECKED (c9), aspect 4, signal log-size spread 0.7, momentum 0.95. Correlation over neurons of log h between step 100 and step 120:

| per-entry signal/noise per step | n = 32 | n = 128 | n = 512 | row-length CV at n = 512 (formula) |
|---|---|---|---|---|
| 0.03 | 0.18 | 0.33 | 0.62 | 0.044 (0.042) |
| 0.1 | 0.73 | 0.90 | 0.97 | 0.155 (0.154) |
| 1.0 | 0.99 | 1.00 | 1.00 | 0.441 (0.441) |
| noise only | 0.11 | 0.12 | 0.13 | CV 0.116 / 0.056 / 0.027 (n = 32 / 128 / 512) |

Real momenta show CV 0.13–0.86 (c6) and neuron-size rank persistence ≥ 0.9 over 100 steps (§20), so they are in the high-signal regime.
CONJECTURE: bigger batches raise the signal-to-noise ratio, so large-batch training passes more of σ into Muon's rows. That would make the row fix more relevant at lab scale, not less.

---

## 4. What the equal-rows fix does geometrically (question 4)

**Plain version.** Equal rows keep every neuron's step direction from Muon and change only its length. The whole step tilts by a small angle set by how uneven Muon's rows were.
Example: Muon row-length CV 0.4 → the equal-rows step has cosine 0.93 with Muon's step (22°). CV 0.2 → 0.98 (11°).

### 4.1 Same directions, new lengths (PROVEN)

R = D·O with D = diag(d_i), d_i = √(n/m)/ℓ_i (this keeps |R|_F = |O|_F = √n). Row i of R is parallel to row i of O, and every row has length √(n/m). By §1.1, R's row i = √(n/m)·A^(−1/2)m_i/|A^(−1/2)m_i|: whiten the gradient rows, then normalize each one. The row space of R equals that of O (R moves neurons within the same input directions).

### 4.2 Exact angle (PROVEN)

⟨O, R⟩ = Σ_i d_i ℓ_i² = √(n/m)·Σ_i ℓ_i. With Σℓ_i² = n:

    cos(O, R) = mean(ℓ)/√(mean(ℓ²)) = 1/√(1 + CV(ℓ)²).

### 4.3 Bounds on R's singular values (PROVEN)

RᵀR = V·(UᵀD²U)·Vᵀ. For a unit vector x, xᵀUᵀD²Ux = Σ_i d_i²(Ux)_i² is a weighted mean of the d_i², because Σ(Ux)_i² = 1. So every singular value of R lies in

    [ℓ_rms/ℓ_max, ℓ_rms/ℓ_min],   ℓ_rms = √(n/m).

Since ℓ_max ≤ 1, no direction is ever shrunk below √(n/m) (0.5 at aspect 4). The upper bound can be large, because it is set by the weakest neuron.

### 4.4 Typical distortion (derivation assumes row lengths independent of U's row directions; CHECKED, within ~25%)

RᵀR − I = Uᵀ(D² − I)U, since UᵀU = I. If U's rows behave like a random frame independent of D, the diagonal terms dominate, giving |RᵀR − I|_F/√n ≈ √(mean(E²)/a) with E_i = d_i² − 1 ≈ −2δ_i (δ_i = relative deviation of ℓ_i). So

    |RᵀR − I|_F/√n ≈ 2·CV(ℓ)/√(m/n).

CHECKED (c6), lognormal synthetic and real momentum snapshots from those small runs (`snaps/*.pt`, not included; c6 skips them when absent):

| matrix | CV(ℓ) | cos measured = 1/√(1+CV²) | singular values of R | bound | |RᵀR−I|_F/√n (≈ 2CV/√a) |
|---|---|---|---|---|---|
| lognormal a=4 σ=0.25 | 0.181 | 0.98393 = 0.98393 | [0.81, 1.19] | [0.67, 2.17] | 0.179 (0.181) |
| lognormal a=4 σ=1 | 0.609 | 0.85410 = 0.85410 | [0.52, 1.67] | [0.51, 27] | 0.631 (0.609) |
| lognormal a=2 σ=1 | 0.379 | 0.93503 = 0.93503 | [0.71, 1.73] | [0.71, 9.0] | 0.533 (0.536) |
| lognormal a=8 σ=1 | 0.798 | 0.78144 = 0.78144 | [0.41, 1.64] | [0.36, 54] | 0.612 (0.564) |
| mlpx ReLU² 1024×256 | 0.398 | 0.92923 = 0.92923 | [0.66, 1.44] | [0.56, 114] | 0.351 (0.398) |
| mlpx ReLU³ | 0.862 | 0.75745 = 0.75745 | [0.53, 2.32] | [0.51, 1e15] | 0.818 (0.862) |
| mlpx SwiGLU 2048×256 | 0.440 | 0.91527 = 0.91527 | [0.67, 1.35] | [0.46, 7.9] | 0.325 (0.311) |
| mlpx SiLU | 0.231 | 0.97442 = 0.97442 | [0.80, 1.27] | [0.67, 3.2] | 0.219 (0.231) |
| mlpx GELU | 0.186 | 0.98322 = 0.98322 | [0.82, 1.21] | [0.70, 2.4] | 0.179 (0.186) |
| mlpx tanh | 0.130 | 0.99163 = 0.99163 | [0.87, 1.31] | [0.75, 5.7] | 0.124 (0.130) |
| lm SwiGLU gate 336×128 | 0.346 | 0.94491 = 0.94491 | [0.73, 1.46] | [0.67, 18.6] | 0.344 (0.427) |

The lower bound is nearly tight. The upper bound is loose: only a few rows are very short, and their boost is diluted across many directions.
**When the cost is small:** the change of step direction depends only on CV(ℓ): CV ≤ 0.4 means cos ≥ 0.93. The fix gives up some of Muon's "every singular direction at 1" property in exchange. The typical singular value moves by about 2·CV/√(m/n): ±0.2 at CV 0.2 and aspect 4.

---

## 5. Wide matrices and MoE (question 5)

**Plain version.** On a wide matrix (fewer rows than columns), Muon's rows are exactly equal and the columns become uneven instead. Which side is "a hidden neuron" depends on whether the layer expands or contracts.
Example: for a 2×4 matrix, msign gives two rows of length exactly 1. The four columns share a budget of 2.

### 5.1 Identities (PROVEN)

For wide M (m < n, full row rank): O = (MMᵀ)^(−1/2) M, so OOᵀ = I_m and every row has length exactly 1. OᵀO is the projector onto M's row space, so column j has squared length g_jᵀ(MMᵀ)⁻¹g_j (the leverage of column j), and these sum to m. Everything in §1–4 holds with rows ↔ columns.
For tall M, OᵀO = I_n, so every column has length exactly 1.

### 5.2 Where the hidden neurons are (PROVEN)

| layer | shape (out × in) | hidden neurons are | Muon's step per hidden neuron |
|---|---|---|---|
| dense FFN up/gate (d_ff = 2.7–4 × d) | tall | rows | uneven (leverage) |
| dense FFN down | wide | columns | uneven (column leverage) |
| fine-grained MoE expert up/gate (d_ff_expert < d_model) | wide | rows | exactly equal, length 1 |
| fine-grained MoE expert down | tall | columns | exactly equal, length 1 |

CHECKED (c7, shapes scaled 1/16): MoE up 128×448 has row CV 0.0000 (all rows 1.0000). MoE down 448×128 has column CV 0.0000. The dense FFN shapes show CV 0.59–0.64 on the hidden-neuron side.

Consequences:
- In fine-grained MoE experts (DeepSeek-V4: expert 3072 ↔ 7168; Step 5: 1536 ↔ 4096; §11), Muon already gives every hidden neuron the same step size on both projections. The feedback loop of §2–3 (through φ of the hidden neuron) has no per-neuron step difference to amplify. (PROVEN for the step sizes; that the loop is then absent is CONJECTURE.)
- In those experts, the tall matrix is the down projection, and its uneven rows are residual-stream dimensions. These do not pass through the expert's activation. So the mechanism of §2 does not apply to them directly. The row fix there would equalize residual dimensions, a different effect. (CONJECTURE: predicts a smaller gain from equal rows in fine-grained MoE experts than in dense ReLU²/SwiGLU FFNs. Testable on a tiny MoE.)
- In a dense FFN, hidden neurons are uneven on both sides. Wide-notes found that evening the down-projection columns adds nothing on top of evening up-projection rows. Fits §2: the up projection's rows sit before the activation, where the self-amplifying gate acts. The down projection's columns sit after it. (CONJECTURE)
- Square matrices (d × d attention projections, Kimi K3's near-square latent experts, aspect 1.17) have nearly equal rows and columns (§1.5, a → 1), so there is nothing to fix.

---

## 6. Downstream coupling β (open item from §2.2)

**Plain version.** A neuron's input weights and output weights are tied. Under plain gradient descent they grow together by an exact rule.
Example: a ReLU² neuron starts with |w1|² = 1 and |w2|² = 0.5. Gradient descent keeps |w1|² − 2|w2|² = 0 forever. If |w1| grows to 10, then |w2|² = 50, so |w2| = 7.07 = |w1|/√2.

### 6.1 Balance identity (PROVEN; CHECKED c13a to 6·10⁻¹⁶)

For ReLU^p, z·φ'(z) = p·φ(z) (Euler, homogeneity). Let neuron i have input row w1_i, output column w2_i, preactivation z_it = w1_i·x_t, and backward error δ_t at the output. Then

    w1_i·g1_i = Σ_t (w2_i·δ_t) φ'(z_it) z_it = p Σ_t (w2_i·δ_t) φ(z_it) = p·(w2_i·g2_i).

This holds for every neuron, any data and any loss. Under gradient flow d|w1_i|²/dt = −2 w1_i·g1_i and d|w2_i|²/dt = −2 w2_i·g2_i, so |w1_i|² − p|w2_i|² stays constant. Decoupled weight decay shrinks this difference toward 0, so it keeps the balance too. From small init, |w2_i| ≈ |w1_i|/√p for every neuron. The error reaching neuron i is w2_i·δ ∝ r_i, so **β = 1 under gradient descent**. SiLU breaks the identity (relative error 0.33 in c13a), because z·φ' ≠ p·φ.

With β = 1, §2.2 gives el = (1 + p − 1)/p = 1 for every ReLU^p. The measured values are 0.57–0.88, so β < 1 in the Muon runs. Muon orthogonalizes W1 and W2 separately, so the step on w1_i is no longer tied to the step on w2_i. CONJECTURE: β < 1 measures how far Muon decouples a neuron's input and output weights.

### 6.2 Snapshot proxy (CHECKED c13b; mapping to el is CONJECTURE)

A direct β needs the trained W2 and δ. Only momenta were saved, and producing weights needs a training run, which these notes exclude. The saved momenta give a proxy. Row i of the l1 momentum is Σ_t (w2_i·δ_t) φ'(z_it) x_t. Column i of the l2 momentum is Σ_t δ_t φ(z_it), whose size grows with neuron i's activation size. If |l2 column i| ∝ activation size, the log-log slope of |l1 row i| on |l2 column i| is el. Pairs are from the same seed and settings at step 750 (`snaps/mlpx_*.pt` and `snaps/wide_mlpx_*.pt`):

| activation | slope (proxy el) | Pearson | Spearman | el measured directly (§20) | implied β = p·el − (p − 1) |
|---|---|---|---|---|---|
| ReLU² | 0.846 | 0.969 | 0.933 | 0.78 | 0.69 (fit from ReLU: 0.57; GD: 1) |
| GELU | 0.665 | 0.898 | 0.884 | 0.55 | not homogeneous |
| SwiGLU up rows | 0.859 | 0.962 | 0.953 | not measured | |
| SwiGLU gate rows | 0.868 | 0.968 | 0.962 | not measured | |

- The same neurons dominate both matrices' gradients (rank correlation 0.88–0.96).
- The proxy reads 0.07 (ReLU²) and 0.12 (GELU) above the direct el. Applying ReLU²'s offset to SwiGLU gives el ≈ 0.78, the same as ReLU². CONJECTURE.
- β lies between 0.57 (fit) and 0.69 (proxy), well below gradient descent's 1.

### 6.3 The three predictions of section 10: what math alone gives

All three need training runs, so they are not tested here. Math-only parts:
- (a) GELU and ReLU: q_M = 0.42–0.43 vs SiLU 0.59 and ReLU² 1.08 (§2.3). Both are below the runaway line, so any gain should be at most SiLU's.
- (b) Fine-grained MoE: now exact at full shape (section 7). Hidden neurons already get equal steps, so the hidden-neuron part of the gain should vanish.
- (c) Aspect threshold: ReLU² runaway at a > 2.79; SwiGLU at a > 2.1–2.8 (section 8). The gain should rise steeply across that range.

---

## 7. Fine-grained MoE at full shape (question 5, exact)

**Plain version.** In a DeepSeek-style expert, every hidden neuron gets exactly the same Muon step on all three matrices. The only uneven steps are on residual-stream dimensions in the down projection.
Example: gate 3072×7168 with hidden-neuron gradient sizes spread by σ = 0.7 (a 4× range is common). All 3072 rows of msign come out 1.000000000000.

CHECKED (c14, c14b–d; full shapes, exact msign via eigh in float64, about 20 s each):

| matrix | hidden-neuron side | residual side |
|---|---|---|
| gate or up 3072×7168 (wide) | rows: min = max = 1.000000000000 | columns: CV 0.010, squares sum to 3072.000 |
| down 7168×3072 (tall), residual gradient sizes equal | columns: min = max = 1.000000000000 | rows: CV 0.010 (noise law √((1−n/m)/(2n)) = 0.0096) |
| down, residual sizes spread σ = 0.5 | columns: exactly 1 | rows: CV 0.261 (small-spread rule 0.286) |
| down, 16 residual dims with 10× gradient | columns: exactly 1 | those 16 rows get 0.993, the rest 0.654 |

- **Exact equality (PROVEN).** Wide: OOᵀ = I. Tall: OᵀO = I. Gate and up have hidden neurons as rows of a wide matrix; down has them as columns of a tall matrix. So all three give each hidden neuron a step of exactly 1, whatever the gradient sizes.
- **The uneven rows of down are residual dimensions (PROVEN by shape).** Down maps d_ff → d_model, so its 7168 rows are residual-stream outputs.
- **Saturation (PROVEN, CHECKED).** A residual dimension with 10× the gradient gets only a 1.52× step (0.993 vs 0.654). Its leverage is near 1, and by §1.3 the sensitivity 1 − h is near 0.
- **Can the feedback loop act on those rows?** The §2 loop needs a gate φ' of the neuron's own output. Down's rows are added to the residual stream with no activation after them inside the expert, so the §2.1 gate identity has nothing to act on (PROVEN by architecture). Downstream they pass through RMSNorm, whose Jacobian is (1/rms)(I − x̂x̂ᵀ)·diag(g) with x̂ = x/|x|. For a dimension that dominates x, x̂_j is large and the projection cancels most of its own gradient. So a bigger residual dimension gets a relatively smaller gradient: zero or negative feedback, like tanh. CONJECTURE: no runaway on residual rows. Attention and later layers can still couple dimensions.
- **Caveat 1: rank (CHECKED, c14).** Exact equality needs the momentum to have rank d_ff, that is at least d_ff distinct token gradients per expert inside the momentum window. At 1/16 scale (192×448): rank 64 gives hidden-row CV 0.43, rank 128 gives 0.21, rank 191 gives 0.009, rank 192 gives 0. Below full rank the rows act like a tall d_ff × rank matrix and leverage returns. Large-batch MoE training gives each expert far more tokens than d_ff per step (CONJECTURE about practice).
- **Caveat 2: practical Newton–Schulz** gives row lengths only within [min f, max f] (section 9).
- **Caveat 3: fusing.** A fused [gate; up] is 2·d_ff × d_model. It stays wide while 2·d_ff < d_model (DeepSeek 6144×7168). It turns tall when 2·d_ff > d_model: Kimi K3's latent experts (3072 ↔ 3584) fused would be 6144×3584, aspect 1.71, and leverage returns.

---

## 8. Dense SwiGLU: predicted unevenness and feedback (question 1 and 2 applied)

**Plain version.** A big dense SwiGLU layer should have Muon row-length CV about 0.3–0.4, and its feedback exponent sits at or past the runaway line.
Example: Llama-3 8B up projection 14336×4096 (aspect 3.5). With gradient-row log-sizes spread by σ = 0.6, the formula gives CV 0.385.

### 8.1 Row-length CV (formula PROVEN in the large-width limit; CHECKED c16)

The §1.5 formula needs only the row sizes, so it runs at full shape. Predicted CV of Muon's row lengths:

| matrix (aspect) | σ 0.3 | σ 0.5 | σ 0.6 | σ 0.7 | σ 0.8 |
|---|---|---|---|---|---|
| DeepSeek-V3 dense up 18432×7168 (2.57) | 0.175 | 0.278 | 0.325 | 0.366 | 0.412 |
| Llama-3 8B up 14336×4096 (3.50) | 0.207 | 0.329 | 0.385 | 0.437 | 0.484 |
| fused [gate; up] 36864×7168 (5.14) | 0.236 | 0.381 | 0.447 | 0.512 | 0.571 |
| fused [gate; up] 28672×4096 (7.00) | 0.255 | 0.411 | 0.486 | 0.562 | 0.626 |

- Exact SVD at 1/16 scale, σ 0.7: 1152×448 gives 0.372 (formula 0.372); 896×256 gives 0.433 (formula 0.433).
- Measured σ of SwiGLU momentum rows is 0.60–0.61 (c13, mlpx step 750). σ stayed flat over a 4× width range (§8). CONJECTURE: σ ≈ 0.6 at full size, giving CV 0.33 (DeepSeek-V3 dense) and 0.39 (Llama-3 8B).
- Fusing gate and up into one matrix doubles the aspect and raises CV by about 0.1. Splitting lowers it. Which Muon implementation fuses is a choice, and it changes the unevenness (PROVEN from the formula).
- At σ ≥ 0.5 the small-spread rule (1 − n/m)·σ overestimates (0.43 vs 0.37 at σ 0.7, aspect 2.57). Use the formula.

### 8.2 Feedback exponent (PROVEN for the toy model of §2.3; el for SwiGLU is CONJECTURE)

SwiGLU output = silu(w_g·x)·(w_u·x). Scaling both rows by r (c16, Gaussian inputs) gives a_A = 2.1–2.2 and q₀ = 0.56 / 0.54 / 0.50 at r = 0.5 / 1 / 2, close to ReLU²'s 0.50. With el = 0.78 (§6.2 estimate) or 0.86 (raw proxy), q_M = q₀ + (1 − n/m)·el:

| r | el | q_M at a = 2.57 | a = 3.5 | a = 7 (fused) | runaway when aspect > |
|---|---|---|---|---|---|
| 0.5 | 0.78 / 0.86 | 1.03 / 1.08 | 1.11 / 1.17 | 1.22 / 1.29 | 2.33 / 2.07 |
| 1 | 0.78 / 0.86 | 1.02 / 1.07 | 1.10 / 1.16 | 1.21 / 1.28 | 2.43 / 2.15 |
| 2 | 0.78 / 0.86 | 0.97 / 1.02 | 1.05 / 1.11 | 1.17 / 1.23 | 2.82 / 2.41 |

Threshold: q_M > 1 when a > 1/(1 − (1 − q₀)/el). DeepSeek-V3's dense layers (2.57) sit on the line. Llama-3 8B (3.5) and any fused layout are past it in every row. Prediction (toy level): the equal-rows gain for SwiGLU grows with aspect and is clearly present by aspect 3.5.

---

## 9. Practical Newton–Schulz keeps the exact results on the short side (new; PROVEN bound, CHECKED)

**Plain version.** Muon uses 5 Newton–Schulz steps, not an exact SVD. On the short side (rows of a wide matrix) this can only move lengths within a fixed band. On the long side it matches the exact answer.
Example: if NS maps the singular values into [0.68, 1.13], every row of a wide matrix has length between 0.68 and 1.13.

- **Bound (PROVEN).** NS output = U f(Σ) Vᵀ, where f is the scalar quintic iterated 5 times. For wide M, U is m×m orthogonal, so row i has length² Σ_k U_ik² f(σ_k)², a weighted average of f² with weights summing to 1. So every row length lies in [min f, max f] over M's singular values. Tall: the same for columns.
- **Snapshots (CHECKED c15).** Short-side NS5 lengths lie in [0.72, 1.17] with CV 0.016–0.054, inside f's range. Long-side CV equals the exact value to ±0.005 (correlation ≥ 0.996). So NS5 does not change the dense-FFN picture.
- **Full MoE shape (CHECKED c15).** On gate/up 3072×7168 (σ = 0.7), NS5 hidden-row lengths span [0.39, 1.13], CV 0.12. The smallest normalized singular values (~10⁻³) are not reached in 5 steps (f(10⁻³) = 0.47). Mean step by row-size decile, smallest to largest: 0.98, 1.11, 1.02, 0.89, 0.82, 0.84, 0.86, 0.82, 0.88, 0.98. The correlation with log size is −0.34. The map f oscillates in σ (0.47, 0.86, 1.15, 0.70, 0.68, 0.76, 0.71, 0.75, 0.77, 0.70 at σ = 10⁻³ … 1).
- So NS5 brings back a CV of about 0.1 on MoE hidden neurons. It is not monotone in neuron size, so it gives no positive el for the §2 loop to use. CONJECTURE: harmless for the loop. As matrices grow, σ_min/|M|_F shrinks, so more rows fall in the unconverged region. This part depends on size.

---

## 10. Open, and what would falsify this

- β (§2.2): a direct value needs the trained W2 and the backward error δ, which were not saved; producing them needs a training run (§6.2 gives the snapshot proxy). The proxy reads 0.07–0.12 above the directly measured el where both exist.
- The growth model of §2.3 treats a neuron as one size number. Real neurons also rotate. §16 found that equalizing weight norms does not help, so "size" has to include alignment with high-variance input directions, not only the weight norm. For ReLU² the gradient is linear in w given the gating pattern (g_i = 2E[δ·1(w·x>0)·x xᵀ]·w), which is one step of power iteration. A neuron with a persistently bigger step size does more power-iteration steps and collapses toward the top input directions faster. This would explain the ~10% loss of directions in §18. CONJECTURE.
- Predictions to test (each needs training runs, so not done here; §6.3 gives the math-only parts): (a) GELU and ReLU, 8 seeds: small gains near SiLU's (§2.3 table). (b) A fine-grained MoE expert (d_ff < d_model): smaller gain from equal rows than a dense FFN with the same activation (§5.2). (c) The aspect threshold: for ReLU² the gain should grow sharply between aspect 2 and 4 (q_M crosses 1 near 2.8).
