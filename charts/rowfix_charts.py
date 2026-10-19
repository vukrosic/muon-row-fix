import math, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erf

OUT = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/"
INK, MUTED, GRID, BG = "#1d1d1f", "#6e6e73", "#e5e5ea", "#ffffff"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS", "DejaVu Sans"], "font.size": 14, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

def style(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

def head(fig, t, s):
    fig.text(0.06, 0.94, t, fontsize=22, fontweight="bold", va="top")
    fig.text(0.06, 0.875, s, fontsize=14.5, color=MUTED, va="top")

# ---------- chart 1: slope vs output ----------
z = np.linspace(1e-4, 4, 4000)
sig = 1 / (1 + np.exp(-z))
curves = [
    ("ReLU²", z**2, 2 * z, "rises → apply the fix"),
    ("SwiGLU", z * sig * z, np.sqrt((z * sig)**2 + (sig * (1 + z * (1 - sig)) * z)**2), "rises → apply the fix"),
    ("SiLU", z * sig, sig * (1 + z * (1 - sig)), "rises a little → small gain"),
    ("ReLU", z, np.ones_like(z), "flat → little or no gain"),
    ("tanh", np.tanh(z), 1 - np.tanh(z)**2, "falls → skip the fix"),
]
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150, facecolor=BG)
fig.subplots_adjust(left=0.08, right=0.72, top=0.78, bottom=0.17)
head(fig, "For some activations, a bigger neuron gets a bigger gradient",
     "So it keeps growing: a few neurons take over and the rest learn little. The row fix stops this.")
for (name, phi, d, tag), c in zip(curves, C):
    m = phi <= 2.0
    ax.plot(phi[m], d[m], color=c, lw=2.6)
    dy = {"SiLU": 0.06, "ReLU": -0.06, "tanh": 0.2}.get(name.split('  ')[0], 0)
    ax.text(phi[m][-1] + 0.04, d[m][-1] + dy, f"{name.split('  ')[0]}: {tag}", color=INK, va="center", fontsize=13.5,
            fontweight="bold")
    ax.plot(phi[m][-1], d[m][-1], "o", color=c, ms=7, mec=BG, mew=2)
ax.set_xlim(0, 2); ax.set_ylim(0, 3.0)
ax.set_xlabel("neuron output"); ax.set_ylabel("gradient multiplier")
style(ax)
fig.text(0.06, 0.065, "Gain = lower training loss after the same number of steps.", fontsize=12.5, color=MUTED)
fig.text(0.06, 0.03, "Curves are exact formulas. SwiGLU curve: the gate and up weights of one neuron growing together.",
         fontsize=12.5, color=MUTED)
fig.savefig(OUT + "rowfix_chart_slopes.png", facecolor=BG); plt.close(fig)

# ---------- chart 2: recent open models, where to use the fix ----------
# shapes from each config.json (hidden_size -> intermediate sizes)
from matplotlib.patches import FancyBboxPatch
cols = [
    ("#1baf7a", "USE IT", "on every MLP layer", [
        ("Olmo 3.1 32B", "5120 → 27648, all 64 layers"),
        ("Gemma 4 31B", "5376 → 21504, all 60 layers"),
        ("Qwen3.8 27B", "5120 → 17408, all 64 layers"),
        ("Mistral Medium 3.5", "12288 → 28672, all 88 layers"),
        ("Llama 4 Maverick", "dense layers and experts are tall"),
    ]),
    ("#eda100", "SMALL GAIN", "only a few layers are tall", [
        ("Kimi K3", "1 dense layer of 93"),
        ("GLM-5.3", "3 dense layers of 78"),
        ("MiniMax-M3", "3 dense layers of 60"),
        ("Nemotron 3.5 Super", "shared expert only"),
    ]),
    ("#8e8e93", "SKIP", "every MLP matrix is wide or square", [
        ("DeepSeek-V4 Pro", "experts 7168 → 3072"),
        ("Qwen3.8 2.4T-A95B", "experts 8192 → 2048"),
        ("Mistral Small 4", "experts 4096 → 2048"),
        ("Gemma 4 26B-A4B", "experts 2816 → 704"),
        ("gpt-oss-120b", "experts 2880 → 2880"),
    ]),
]
fig = plt.figure(figsize=(12, 7.6), dpi=150, facecolor=BG)
fig.text(0.06, 0.94, "Where to use the row fix in recent open models", fontsize=22, fontweight="bold", va="top")
fig.text(0.06, 0.875, "Use it on MLP layers with more neurons than inputs (tall), e.g. 5120 → 27648.\n"
         "Where neurons ≤ inputs, Muon already gives every neuron the same step.", fontsize=14.5, color=MUTED, va="top")
W, X0, GAP, TOP, BOT = 0.28, 0.06, 0.03, 0.74, 0.1
for j, (col, head_t, sub, items) in enumerate(cols):
    x = X0 + j * (W + GAP)
    fig.patches.append(FancyBboxPatch((x, BOT), W, TOP - BOT, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc="#f5f5f7", ec="none", zorder=0))
    fig.patches.append(FancyBboxPatch((x, TOP - 0.115), W, 0.115, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fc=col, ec="none", zorder=1))
    fig.text(x + 0.02, TOP - 0.025, head_t, fontsize=20, fontweight="bold", color="white", va="top", zorder=2)
    fig.text(x + 0.02, TOP - 0.075, sub, fontsize=12.5, color="white", va="top", zorder=2)
    for i, (name, det) in enumerate(items):
        y = TOP - 0.165 - i * 0.095
        fig.text(x + 0.02, y, name, fontsize=15, fontweight="bold", color=INK, va="top")
        fig.text(x + 0.02, y - 0.036, det, fontsize=12, color=MUTED, va="top")
fig.text(0.06, 0.035, "MLP shapes from each model's config.json on Hugging Face. All use SwiGLU-type MLPs or ReLU², where the fix applies.\n"
         "Predicted from math, not measured on these models.", fontsize=11, color=MUTED, va="bottom")
fig.savefig(OUT + "rowfix_chart_shape.png", facecolor=BG); plt.close(fig)

# ---------- chart 3: persistent vs redrawn ----------
rng = np.random.default_rng(0)
N, T, eta, q = 1024, 1500, 0.02, 0.5
sg = math.sqrt(math.log(1 + 0.4**2))
def run(persistent):
    x = np.ones(N); f0 = np.exp(sg * rng.standard_normal(N) - sg**2 / 2); out = []
    for t in range(T):
        f = f0 if persistent else np.exp(sg * rng.standard_normal(N) - sg**2 / 2)
        x = x + eta * f * x**q
        out.append(np.log(x).std())
    return np.array(out)
pe, rd = run(True), run(False)
ts = np.arange(1, T + 1)
fig, ax = plt.subplots(figsize=(12, 6.75), dpi=150, facecolor=BG)
fig.subplots_adjust(left=0.08, right=0.80, top=0.78, bottom=0.17)
head(fig, "Same neurons every step vs. different neurons every step",
     "Every neuron gets ±40% step size. Spread of neuron sizes over training (growth model, ReLU²-like feedback).")
ax.plot(ts, pe, color=C[1], lw=2.6); ax.plot(ts, rd, color=C[0], lw=2.6)
ax.text(T + 20, pe[-1], f"same neurons\nevery step: {pe[-1]:.2f}", va="center", fontsize=13.5, fontweight="bold")
ax.text(T + 20, rd[-1] + 0.02, f"different neurons\nevery step: {rd[-1]:.2f}", va="center", fontsize=13.5, fontweight="bold")
ax.set_xlim(0, T); ax.set_ylim(0, max(pe) * 1.1)
ax.set_xlabel("training step"); ax.set_ylabel("spread of neuron sizes (std of log size)")
style(ax)
fig.text(0.06, 0.03, f"Persistent differences add up; redrawn ones average out ({pe[-1] / rd[-1]:.0f}× smaller here). "
         "Muon's differences are the persistent kind.", fontsize=12.5, color=MUTED)
fig.savefig(OUT + "rowfix_chart_persist.png", facecolor=BG); plt.close(fig)
print(pe[-1], rd[-1])
