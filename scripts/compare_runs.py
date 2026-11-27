"""Plot held-out success per loop iteration for several runs.  uv run python scripts/compare_runs.py RUN_ID..."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

runs = sys.argv[1:] or ["demo-heuristic", "demo-codex"]
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for color, run in zip(SERIES, runs):
    h = json.loads((ROOT / "out" / "runs" / run / "history.json").read_text())
    xs, ys = [r["iteration"] for r in h], [r["heldout_success"] * 100 for r in h]
    ax.plot(xs, ys, color=color, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, label=run, zorder=3)
    ax.annotate(f"{run}  {ys[-1]:.0f}%", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
ax.set_ylim(40, 102)
ax.set_xlim(-0.2, max(len(json.loads((ROOT / "out" / "runs" / r / "history.json").read_text())) for r in runs) + 0.8)
ax.set_xlabel("Loop iteration (0 = v0 baseline)", color=MUTED)
ax.set_ylabel("Held-out success rate (%)", color=MUTED)
ax.set_title("Same v0 controller, same held-out seeds: which brain fixes it faster?", loc="left", fontsize=10, color=INK)
ax.legend(frameon=False, loc="lower right", labelcolor=INK)
ax.grid(axis="y", color="#e6e5e0", lw=0.8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#c9c8c2")
ax.tick_params(colors=MUTED)
fig.tight_layout()
out = ROOT / "out" / "compare.png"
fig.savefig(out, facecolor=SURFACE)
print(out)
