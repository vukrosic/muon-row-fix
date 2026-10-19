import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mt
OUT = __import__("os").path.dirname(__import__("os").path.abspath(__file__)) + "/"
INK, MUTED, GRID, BG = "#1d1d1f", "#6e6e73", "#e5e5ea", "#ffffff"
BLUE, GREEN = "#2a78d6", "#1baf7a"
plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS", "DejaVu Sans"], "font.size": 14, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
d = np.load(sys.argv[1])
rng = np.random.default_rng(1)
fig, axs = plt.subplots(1, 2, figsize=(12, 6.75), dpi=150, facecolor=BG, sharey=True)
fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.2, wspace=0.08)
fig.text(0.06, 0.94, "Muon gives bigger steps to neurons with bigger gradients, only on tall layers",
         fontsize=21, fontweight="bold", va="top")
fig.text(0.06, 0.875, "Exact Muon step for every neuron, same kind of gradient, two real layer shapes. Each dot is one neuron.",
         fontsize=14, color=MUTED, va="top")
panels = [("tall", "Qwen3.8 27B MLP:  5120 → 17408 (tall)"), ("wide", "DeepSeek-V4 Pro expert:  7168 → 3072 (wide)")]
for ax, (k, title) in zip(axs, panels):
    s, l = d[k + "_s"], d[k + "_l"]
    i = rng.choice(len(s), 3000, replace=False)
    ax.scatter(s[i], l[i], s=6 if k == "tall" else 9, color=BLUE, alpha=0.35 if k == "tall" else 0.6, lw=0, zorder=3)
    ax.axhline(1, color=GREEN, lw=2.6, zorder=2)
    ax.set_xscale("log"); ax.set_xlim(0.15, 6); ax.set_ylim(0, 2.0)
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.set_xticks([0.25, 0.5, 1, 2, 4]); ax.set_xticklabels(["¼×", "½×", "1×", "2×", "4×"])
    ax.set_title(title, fontsize=14.5, loc="left", color=INK, pad=12)
    ax.set_xlabel("neuron's gradient size (vs. average)")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
axs[0].set_ylabel("neuron's step length (vs. average)")
k = np.polyfit(np.log(d["tall_s"]), np.log(d["tall_l"]), 1)[0]
lt = d["tall_l"]
axs[0].text(0.17, 1.8, f"Muon: 2× bigger gradient → {2**k:.2f}× bigger step", color=BLUE, fontsize=13.5, fontweight="bold")
axs[0].text(0.17, 1.66, f"largest step is {lt.max() / lt.min():.0f}× the smallest", color=INK, fontsize=12.5)
axs[0].text(5.8, 1.06, "the fix: every neuron 1.0", color=GREEN, fontsize=13, fontweight="bold", ha="right")
axs[1].text(0.17, 1.8, "Muon: every neuron gets 1.000 (dots on the line)", color=BLUE, fontsize=13.5, fontweight="bold")
axs[1].text(0.17, 1.66, "already equal, the fix changes nothing", color=INK, fontsize=12.5)
fig.text(0.06, 0.035, "Gradient rows: random directions, sizes spread from about ¼× to 4× the average. Exact Muon step (polar factor), "
         "no training.\nOn the tall layer the step grows on average as gradient^0.71, matching 1 − inputs/neurons = 0.71.",
         fontsize=11, color=MUTED, va="bottom")
fig.savefig(OUT + "rowfix_chart_realshape.png", facecolor=BG)
print(k, lt.min(), lt.max(), d["tall_s"].min(), d["tall_s"].max())
