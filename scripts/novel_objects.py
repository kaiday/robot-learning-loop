"""Show the arm a sequence of unfamiliar objects and scenes; it retries and reanalyzes until it picks each up.

    uv run python scripts/novel_objects.py --brain codex --mode both
    uv run python scripts/novel_objects.py --brain heuristic --tasks puck@row ball@row

--mode library   : remember successes and start similar objects from them
--mode scratch   : every object starts from the cube-trained params (no memory)
--mode both      : run both and compare
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rll.adapt import adapt  # noqa: E402
from rll.brains import CodexBrain, HeuristicBrain  # noqa: E402
from rll.library import SuccessLibrary  # noqa: E402
from rll.objects import task  # noqa: E402
from rll.runner import BatchRunner  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ["puck@row", "ball@row", "bar@row", "heavy_ball@row", "slippery_box@row", "cube@row",
                 "heavy_ball@clean", "wide_can@row"]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_sequence(mode: str, tasks: list[str], brain, runner: BatchRunner, start: dict, out: Path, args) -> list[dict]:
    library = SuccessLibrary(out / mode / "library.json") if mode == "library" else None
    rows = []
    for spec in tasks:
        r = adapt(task(spec), brain, runner, start, out / mode / spec.replace("@", "_"), library=library,
                  trials=args.trials, k=args.k, max_rounds=args.max_rounds, log=log)
        rows.append(r.to_dict())
        (out / mode / "results.json").write_text(json.dumps(rows, indent=2))
    return rows


def chart(results: dict[str, list[dict]], path: Path) -> None:
    blue, orange, ink, muted, surface = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
    modes = list(results)
    tasks = [r["task"] for r in results[modes[0]]]
    x = np.arange(len(tasks))
    w = 0.8 / len(modes)
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    for i, (mode, color) in enumerate(zip(modes, [blue, orange])):
        rows = results[mode]
        vals = [r["wall_s"] for r in rows]
        bars = ax.bar(x + (i - (len(modes) - 1) / 2) * w, vals, w * 0.92, color=color, label=mode, zorder=3)
        for b, r in zip(bars, rows):
            label = f"{r['rounds_used']}r" + ("" if r["success"] else " x")
            ax.annotate(label, (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8, color=ink)
    ax.set_xticks(x, tasks, rotation=20, ha="right", fontsize=9, color=ink)
    ax.set_ylabel("Seconds until reliable pick", color=muted)
    ax.set_title("Unfamiliar objects: time to learn each one (labels = retry rounds, x = not solved)",
                 loc="left", fontsize=10, color=ink)
    ax.legend(frameon=False, labelcolor=ink)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def table(results: dict[str, list[dict]]) -> str:
    lines = ["| mode | task | start from | first try | final | validation | rounds | sims | brain s | sim s | total s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for mode, rows in results.items():
        for r in rows:
            v = "-" if r["validation"] is None else f"{r['validation']:.0%}"
            lines.append(f"| {mode} | {r['task']} | {r['start_from']} | {r['first_try']:.0%} | {r['final']:.0%} | {v} | "
                         f"{r['rounds_used']} | {r['sims']} | {r['brain_s']} | {r['sim_s']} | {r['wall_s']} |")
        ok = [r for r in rows if r["success"]]
        lines.append(f"| **{mode} total** | {len(ok)}/{len(rows)} solved | | | | | "
                     f"{sum(r['rounds_used'] for r in rows)} | {sum(r['sims'] for r in rows)} | "
                     f"{sum(r['brain_s'] for r in rows):.0f} | {sum(r['sim_s'] for r in rows):.0f} | "
                     f"{sum(r['wall_s'] for r in rows):.0f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", choices=["heuristic", "codex"], default="heuristic")
    ap.add_argument("--mode", choices=["library", "scratch", "both"], default="both")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--trials", type=int, default=8, help="trials per hypothesis")
    ap.add_argument("--k", type=int, default=4, help="hypotheses tested in parallel per round")
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--start", type=Path, default=ROOT / "policy" / "params_cube_tuned.json")
    ap.add_argument("--no-images", action="store_true", help="don't send keyframes to Codex (faster)")
    ap.add_argument("--run-id")
    a = ap.parse_args()

    run_id = a.run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{a.brain}"
    out = ROOT / "out" / "novel" / run_id
    out.mkdir(parents=True, exist_ok=True)
    brain = CodexBrain(images=not a.no_images) if a.brain == "codex" else HeuristicBrain()
    start = json.loads(a.start.read_text())
    modes = ["library", "scratch"] if a.mode == "both" else [a.mode]
    log(f"run {run_id}: brain={brain.name} modes={modes} tasks={a.tasks}")
    t0 = time.time()
    results = {}
    with BatchRunner() as runner:
        log(f"{runner.workers} parallel simulators ready in {time.time() - t0:.1f}s")
        for mode in modes:
            results[mode] = run_sequence(mode, a.tasks, brain, runner, start, out, a)
    chart(results, out / "time_to_learn.png")
    md = table(results)
    (out / "summary.md").write_text(md)
    print(md)
    log(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
