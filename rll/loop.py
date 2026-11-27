"""The learning loop: Run -> Capture -> Find failure -> Fix -> Run again (gate) -> repeat.

    uv run python -m rll.loop --brain heuristic --iters 4
    uv run python -m rll.loop --brain codex --iters 3
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from policy.controller import PARAMS_PATH, load_params, validate_params  # noqa: E402
from rll.analyze import build_report, report_markdown, success_rate, summary_table  # noqa: E402
from rll.brains import CodexBrain, HeuristicBrain  # noqa: E402
from rll.env import PickPlaceEnv  # noqa: E402
from rll.gate import run_gate  # noqa: E402
from rll.replay import replay_failure  # noqa: E402
from rll.runner import run_batch  # noqa: E402
from rll.storage import get_storage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_SEED_BASE = 900_000  # held-out seeds, never used for training batches


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_loop(
    brain_name: str = "heuristic",
    iters: int = 4,
    n_train: int = 60,
    n_eval: int = 100,
    n_replays: int = 2,
    run_id: str | None = None,
    params: dict | None = None,
    out_root: Path = ROOT / "out" / "runs",
    promote: bool = False,
) -> dict:
    run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{brain_name}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    storage = get_storage()
    brain = CodexBrain() if brain_name == "codex" else HeuristicBrain()
    params = validate_params(params or load_params())
    eval_seeds = list(range(EVAL_SEED_BASE, EVAL_SEED_BASE + n_eval))
    env = PickPlaceEnv()

    log(f"run {run_id}: brain={brain.name} storage={storage.name} eval_seeds={n_eval}")
    base_eval = run_batch(eval_seeds, params)
    history = [{"iteration": 0, "version": 0, "heldout_success": success_rate(base_eval), "params": params,
                "accepted": True, "changes": {}, "diagnosis": "baseline"}]
    (run_dir / "params_v0.json").write_text(json.dumps(params, indent=2))
    log(f"v0 held-out success {success_rate(base_eval):.0%}")
    version = 0

    for it in range(1, iters + 1):
        it_dir = run_dir / f"iter_{it:02d}"
        t0 = time.time()
        # 1. RUN + 2. CAPTURE on fresh training seeds
        train_seeds = list(range(it * 10_000, it * 10_000 + n_train))
        train = run_batch(train_seeds, params)
        it_dir.mkdir(parents=True, exist_ok=True)
        summary_table(train).to_parquet(it_dir / "train_summary.parquet", index=False)

        # 3. FIND THE FAILURE
        report = build_report(train, params)
        (it_dir / "report.json").write_text(json.dumps(report, indent=2))
        (it_dir / "report.md").write_text(report_markdown(report))
        log(f"iter {it}: train success {report['success_rate']:.0%}, failures {report['failure_counts']}")
        failed = [r for r in train if not r.success]
        if not failed:
            log("no failures left in training batch, stopping")
            break
        images = []
        for kind in report["failures"]:  # one replay per failure type, most common first
            if len(images) >= n_replays:
                break
            seed = report["failures"][kind]["examples"][0]["seed"]
            _, png = replay_failure(env, seed, params, it_dir / "failures")
            if png:
                images.append(png)

        # 4. FIX
        proposal = brain.propose(report, params, it_dir / "brain", images)
        cand = proposal.apply(params)
        (it_dir / "proposal.json").write_text(json.dumps(proposal.to_dict(), indent=2))
        log(f"iter {it}: {brain.name} proposes {proposal.changes}")
        log(f"   diagnosis: {proposal.diagnosis[:300]}")

        # 5. RUN AGAIN: gate on held-out seeds
        gate, cand_eval = run_gate(base_eval, cand, eval_seeds, [r.cfg.seed for r in failed])
        (it_dir / "gate.json").write_text(json.dumps(gate.to_dict(), indent=2))
        log(f"iter {it}: gate {'ACCEPT' if gate.accepted else 'REJECT'} ({gate.reason}); "
            f"retried training failures fixed {gate.retry_fixed}/{gate.retry_total}")
        if isinstance(brain, CodexBrain):
            brain.remember(it, proposal.changes, gate.base_rate, gate.cand_rate, gate.accepted)
        if gate.accepted:
            params, base_eval, version = cand, cand_eval, version + 1
            (run_dir / f"params_v{version}.json").write_text(json.dumps(params, indent=2))

        record = {
            "iteration": it, "version": version, "heldout_success": success_rate(base_eval),
            "candidate_success": gate.cand_rate, "train_success": report["success_rate"],
            "failure_counts": report["failure_counts"], "accepted": gate.accepted, "changes": proposal.changes,
            "diagnosis": proposal.diagnosis, "brain": brain.name, "seconds": round(time.time() - t0, 1),
            "params": params,
        }
        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        plot_history(history, run_dir / "success_curve.png")
        storage.record_iteration(run_id, record)
        n = storage.upload_dir(it_dir, f"runs/{run_id}/iter_{it:02d}")
        if n:
            log(f"iter {it}: uploaded {n} files to {storage.name}")

    for f in ("history.json", "success_curve.png", f"params_v{version}.json"):
        if (run_dir / f).exists():
            storage.put_file(f"runs/{run_id}/{f}", run_dir / f)
    if promote:
        PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n")
        log(f"promoted params_v{version} to {PARAMS_PATH}")
    log(f"done: held-out success {history[0]['heldout_success']:.0%} -> {history[-1]['heldout_success']:.0%} "
        f"({version} accepted fixes). Results in {run_dir}")
    return {"run_id": run_id, "run_dir": str(run_dir), "history": history}


def plot_history(history: list[dict], path: Path) -> None:
    blue, ink, muted, surface = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"
    xs = [h["iteration"] for h in history]
    ys = [h["heldout_success"] * 100 for h in history]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot(xs, ys, color=blue, lw=2, marker="o", ms=7, mec=surface, mew=2, zorder=3)
    for h, y in zip(history, ys):
        if h["iteration"] > 0 and not h["accepted"]:
            cand = h["candidate_success"] * 100
            ax.plot([h["iteration"]], [cand], marker="x", ms=7, color=muted, zorder=2)
        ax.annotate(f"{y:.0f}%", (h["iteration"], y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=ink)
    ax.set_ylim(0, 105)
    ax.set_xticks(xs)
    ax.set_xlabel("Loop iteration", color=muted)
    ax.set_ylabel("Held-out success rate (%)", color=muted)
    ax.set_title("Pick-and-place success on held-out seeds  (x = candidate rejected by gate)",
                 loc="left", fontsize=10, color=ink)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--brain", choices=["heuristic", "codex"], default="heuristic")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--train", type=int, default=60, help="training episodes per iteration")
    ap.add_argument("--eval", type=int, default=100, help="held-out episodes for the gate")
    ap.add_argument("--params", type=Path, help="start from this params file (default policy/params.json)")
    ap.add_argument("--run-id")
    ap.add_argument("--promote", action="store_true", help="write the final params back to policy/params.json")
    a = ap.parse_args()
    params = json.loads(a.params.read_text()) if a.params else None
    run_loop(a.brain, a.iters, a.train, a.eval, run_id=a.run_id, params=params, promote=a.promote)


if __name__ == "__main__":
    main()
