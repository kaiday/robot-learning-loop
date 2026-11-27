"""Meet an unfamiliar object: attempt -> find failure -> K hypotheses tested at once -> keep best -> repeat.

When the object is picked reliably, the winning params and a recording of a successful run go into the
success library, so a similar object later starts from what worked.
"""
from __future__ import annotations

import json
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from policy.controller import validate_params
from rll.analyze import HIDDEN, build_report, success_rate
from rll.env import PickPlaceEnv
from rll.library import Memory, SuccessLibrary
from rll.objects import Task, sample_trial
from rll.replay import KEY_OFFSET, RENDER_EVERY
from rll.runner import BatchRunner, EpisodeResult, run_episode, save_episode
from rll.visual import save_grid


@dataclass
class Candidate:
    name: str
    changes: dict
    success: float
    failures: dict
    rationale: str = ""


@dataclass
class Round:
    round: int
    candidates: list[Candidate]
    kept: str
    success: float  # success of the params in use after this round
    brain_s: float = 0.0
    sim_s: float = 0.0
    render_s: float = 0.0
    diagnosis: str = ""
    grid: str = ""


@dataclass
class AdaptResult:
    task: str
    start_from: str
    match_distance: float | None
    success: bool
    first_try: float
    final: float
    validation: float | None
    rounds_used: int
    sims: int
    wall_s: float
    brain_s: float
    sim_s: float
    params: dict
    rounds: list[Round] = field(default_factory=list)
    error: str = ""  # e.g. the brain became unavailable mid-way

    def to_dict(self) -> dict:
        return asdict(self)


def trial_seeds(t: Task, n: int, salt: int = 0) -> list[int]:
    base = (zlib.crc32(t.name.encode()) % 50_000) * 100 + 1_000_000 + salt * 10_000_000
    return [base + i for i in range(n)]


def _failures(results: list[EpisodeResult]) -> dict:
    return dict(Counter(r.failure["type"] for r in results if r.failure))


def _show_step(r: EpisodeResult) -> int:
    """Frame that tells the story: the failure moment, or the object held highest for a success."""
    if r.failure:
        return r.failure["step"]
    return int(r.steps.obj_z.idxmax()) if not r.steps.empty else 0


def _render_tiles(runner: BatchRunner, picks: list[tuple[EpisodeResult, dict, str]]) -> list[tuple[np.ndarray, bool, str]]:
    jobs = [(r.cfg, p, cap, {_show_step(r)}) for r, p, cap in picks]
    out = []
    for (r, _, cap), rr in zip(picks, runner.run(jobs)):
        frame = rr.frames[0] if rr.frames else np.zeros((352, 480, 3), np.uint8)
        out.append((frame, r.success, cap))
    return out


def _keyframe_images(runner: BatchRunner, results: list[EpisodeResult], params: dict, out_dir: Path, n: int = 2) -> list[Path]:
    """Before / at / after strips for up to n failure types, rendered in parallel."""
    seen, picks = set(), []
    for r in results:
        if r.failure and r.failure["type"] not in seen:
            seen.add(r.failure["type"])
            picks.append(r)
        if len(picks) >= n:
            break
    jobs = []
    for r in picks:
        s, last = r.failure["step"], len(r.steps) - 1
        jobs.append((r.cfg, params, r.failure["type"], {max(0, s - KEY_OFFSET), s, min(last, s + KEY_OFFSET)}))
    paths = []
    for r, rr in zip(picks, runner.run(jobs)):
        if rr.frames:
            p = out_dir / f"keyframes_{r.failure['type']}.png"
            iio.imwrite(p, np.concatenate(rr.frames, axis=1))
            paths.append(p)
    return paths


def adapt(
    t: Task,
    brain,
    runner: BatchRunner,
    start_params: dict,
    out_dir: Path,
    library: SuccessLibrary | None = None,
    trials: int = 8,
    k: int = 4,
    max_rounds: int = 4,
    threshold: float = 0.875,
    log=print,
) -> AdaptResult:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = t.obj.features()
    hit = library.lookup(features, t.scene) if library is not None else None
    params = validate_params(hit[0].params if hit else start_params)
    start_from = hit[0].task if hit else "cube-trained params"
    log(f"[{t.name}] unfamiliar object {features}, scene={t.scene}; starting from {start_from}"
        + (f" (distance {hit[1]:.2f})" if hit else ""))

    cfgs = [sample_trial(t, s) for s in trial_seeds(t, trials)]
    ts = time.time()
    results = runner.run([(c, params, "current") for c in cfgs])
    sim_s, brain_s, sims = time.time() - ts, 0.0, trials
    rate = first_try = success_rate(results)
    log(f"[{t.name}] first try: {rate:.0%} {_failures(results)}")
    tiles = _render_tiles(runner, [(r, params, f"trial {i + 1}: {'ok' if r.success else r.failure['type']}")
                                   for i, r in enumerate(results)])
    save_grid(tiles, out_dir / "round_0.png", cols=4, title=f"{t.name}: first try {rate:.0%} ({start_from})")

    rounds: list[Round] = []
    tried: list[dict] = []
    rnd, error = 0, ""
    while rate < threshold and rnd < max_rounds:
        rnd += 1
        rdir = out_dir / f"round_{rnd}"
        rdir.mkdir(exist_ok=True)
        report = build_report(results, params, hidden=HIDDEN, context=t.context())
        tr = time.time()
        images = _keyframe_images(runner, results, params, rdir)
        render_s = time.time() - tr

        tb = time.time()
        try:
            hyps = brain.propose_many(report, params, rdir / "brain", images, k=k, attempts=tried)
        except Exception as exc:  # brain unavailable (quota, network): stop this object, keep what we have
            error = str(exc)
            log(f"[{t.name}] round {rnd}: brain failed: {error}")
            rnd -= 1
            break
        b_s = time.time() - tb
        brain_s += b_s

        ts = time.time()
        cand_params = [h.apply(params) for h in hyps]
        tags = [f"{i}:{h.name or i}" for i, h in enumerate(hyps)]
        all_res = runner.run([(c, cp, tag) for tag, cp in zip(tags, cand_params) for c in cfgs])
        s_s = time.time() - ts
        sim_s += s_s
        sims += len(all_res)

        cands, by_tag = [], {}
        for tag, h in zip(tags, hyps):
            rs = [r for r in all_res if r.tag == tag]
            by_tag[tag] = rs
            cands.append(Candidate(h.name or tag, h.changes, success_rate(rs), _failures(rs), h.expected_effect))
            tried.append({"round": rnd, "name": h.name or tag, "changes": h.changes, "success": success_rate(rs)})
        best = max(range(len(cands)), key=lambda i: cands[i].success)
        kept = "current"
        if cands[best].success > rate:
            params, results, rate, kept = cand_params[best], by_tag[tags[best]], cands[best].success, cands[best].name

        tr = time.time()
        picks = [(by_tag[tag][0], cp, f"{c.name}: {c.success:.0%}") for tag, cp, c in zip(tags, cand_params, cands)]
        grid_path = save_grid(_render_tiles(runner, picks), out_dir / f"round_{rnd}.png", cols=min(4, len(picks)),
                              title=f"{t.name} round {rnd}: {len(hyps)} hypotheses x {trials} trials in parallel")
        render_s += time.time() - tr

        rounds.append(Round(rnd, cands, kept, rate, round(b_s, 1), round(s_s, 1), round(render_s, 1),
                            hyps[0].diagnosis if hyps else "", str(grid_path)))
        log(f"[{t.name}] round {rnd}: brain {b_s:.0f}s, {len(all_res)} sims {s_s:.1f}s -> "
            + ", ".join(f"{c.name} {c.success:.0%}" for c in cands) + f" | kept {kept} ({rate:.0%})")

    validation = None
    if rate >= threshold:
        val_cfgs = [sample_trial(t, s) for s in trial_seeds(t, trials, salt=1)]
        ts = time.time()
        val = runner.run([(c, params, "validation") for c in val_cfgs])
        sim_s += time.time() - ts
        sims += len(val)
        validation = success_rate(val)
        log(f"[{t.name}] validation on {trials} fresh trials: {validation:.0%}")

    success = rate >= threshold and (validation or 0) >= 0.75
    wall_s = time.time() - t0  # learning time; saving the demo recording below is excluded
    if success and library is not None:
        win = next(r for r in results if r.success)
        rec = run_episode(PickPlaceEnv(), win.cfg, params, render_every=RENDER_EVERY)
        rec_dir = save_episode(rec, library.path.parent / t.name, params)
        library.save(Memory(t.name, features, t.scene, params, rate, trials, rnd, str(rec_dir)))
        log(f"[{t.name}] saved to success library ({len(library)} memories), recording {rec_dir}")

    res = AdaptResult(
        task=t.name, start_from=start_from, match_distance=round(hit[1], 3) if hit else None, success=success,
        first_try=first_try, final=rate, validation=validation, rounds_used=rnd, sims=sims,
        wall_s=round(wall_s, 1), brain_s=round(brain_s, 1), sim_s=round(sim_s, 1), params=params,
        rounds=rounds, error=error,
    )
    (out_dir / "result.json").write_text(json.dumps(res.to_dict(), indent=2))
    return res
