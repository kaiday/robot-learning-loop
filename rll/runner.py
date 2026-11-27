"""Run episodes and capture everything the robot saw and did, step by step."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from policy.controller import ScriptedPickPlace
from rll.env import CONTROL_DT, TARGET_TOL, EpisodeConfig, PickPlaceEnv, sample_config

MAX_TIME_S = 15.0
FRAME_W, FRAME_H = 480, 352  # multiples of 16 for video codecs


@dataclass
class EpisodeResult:
    cfg: EpisodeConfig
    success: bool
    steps: pd.DataFrame
    final_dist: float
    failure: dict | None = None  # filled in by rll.detect
    frames: list[np.ndarray] = field(default_factory=list)
    frame_steps: list[int] = field(default_factory=list)
    tag: str = ""  # e.g. which hypothesis produced this episode

    def summary(self) -> dict:
        cfg = {k: v for k, v in self.cfg.to_dict().items() if k not in ("seed", "clutter")}
        return {
            "seed": self.cfg.seed,
            "tag": self.tag,
            "success": self.success,
            "final_dist": round(self.final_dist, 4),
            "failure_type": self.failure["type"] if self.failure else None,
            "failure_phase": self.failure["phase"] if self.failure else None,
            "failure_t": self.failure["t"] if self.failure else None,
            **{f"cfg_{k}": v for k, v in cfg.items()},
            "cfg_half_height": self.cfg.half_height,
            "cfg_n_clutter": len(self.cfg.clutter),
        }


def run_episode(
    env: PickPlaceEnv,
    cfg: EpisodeConfig | int,
    params: dict,
    render_every: int = 0,
    render_steps: set[int] | None = None,
) -> EpisodeResult:
    """One attempt. Render every `render_every` steps (video) and/or exactly at `render_steps` (keyframes)."""
    cfg = sample_config(cfg) if isinstance(cfg, int) else cfg
    env.reset(cfg)
    ctl = ScriptedPickPlace(env, params)
    rows, frames, frame_steps = [], [], []
    renderer = mujoco.Renderer(env.model, FRAME_H, FRAME_W) if (render_every or render_steps) else None
    cam = camera_for(cfg)
    step = 0
    while env.data.time < MAX_TIME_S:
        phase = ctl.phase
        if ctl.done:
            break
        ctrl = ctl.act(CONTROL_DT)
        env.step(ctrl)
        c = env.contacts()
        obj, tcp = env.obj_pos, env.tcp_pos
        rows.append(
            {
                "step": step,
                "t": round(env.data.time, 4),
                "phase": phase,
                "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
                "obj_x": obj[0], "obj_y": obj[1], "obj_z": obj[2],
                "obj_yaw": env.obj_yaw,
                "finger_width": env.finger_width,
                "gripper_cmd": float(ctrl[7]),
                "contact_left": c["left"], "contact_right": c["right"],
                "contact_floor": c["floor"], "contact_arm": c["arm"], "contact_clutter": c["clutter"],
            }
        )
        if renderer is not None and ((render_every and step % render_every == 0) or (render_steps and step in render_steps)):
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
            frame_steps.append(step)
        step += 1
    if renderer is not None:
        renderer.close()

    final = env.obj_pos
    dist = float(np.linalg.norm(final[:2] - env.target_pos[:2]))
    resting = final[2] < cfg.half_height + 0.01
    success = bool(ctl.done and dist < TARGET_TOL and resting)
    return EpisodeResult(cfg=cfg, success=success, steps=pd.DataFrame(rows), final_dist=dist,
                         frames=frames, frame_steps=frame_steps)


def camera_for(cfg: EpisodeConfig) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [(cfg.obj_x + cfg.target_x) / 2, (cfg.obj_y + cfg.target_y) / 2, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 0.9, 150, -30
    return cam


def save_episode(result: EpisodeResult, out_dir: Path, params: dict) -> Path:
    """Write steps.parquet, meta.json and (if rendered as video) video.mp4 for one episode."""
    ep = Path(out_dir) / f"ep_{result.cfg.seed:06d}"
    ep.mkdir(parents=True, exist_ok=True)
    result.steps.to_parquet(ep / "steps.parquet", index=False)
    meta = {**result.summary(), "config": result.cfg.to_dict(), "params": params, "failure": result.failure}
    (ep / "meta.json").write_text(json.dumps(meta, indent=2))
    if len(result.frames) > 10:
        import imageio.v3 as iio

        iio.imwrite(ep / "video.mp4", np.stack(result.frames), fps=int(1 / CONTROL_DT / 2), codec="libx264")
    return ep


# -- parallel batches -------------------------------------------------------------------
_worker_env: PickPlaceEnv | None = None


def _init_worker() -> None:
    global _worker_env
    _worker_env = PickPlaceEnv()


def _run_job(job: tuple) -> EpisodeResult:
    """job = (cfg_or_seed, params, tag) or (cfg_or_seed, params, tag, render_steps)."""
    from rll.detect import detect_failure

    cfg, params, tag = job[:3]
    render_steps = job[3] if len(job) > 3 else None
    assert _worker_env is not None
    res = run_episode(_worker_env, cfg, params, render_steps=render_steps)
    res.failure = detect_failure(res)
    res.tag = tag
    return res


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


class BatchRunner:
    """A pool of simulator processes that stays alive between batches (no re-spawn per round)."""

    def __init__(self, workers: int | None = None) -> None:
        self.workers = workers or default_workers()
        self.pool = mp.get_context("spawn").Pool(self.workers, initializer=_init_worker) if self.workers > 1 else None
        if self.pool is None:
            _init_worker()

    def run(self, jobs: list[tuple[EpisodeConfig | int, dict, str]]) -> list[EpisodeResult]:
        """Results come back in job order, with failures labelled."""
        if self.pool is None:
            return [_run_job(j) for j in jobs]
        return self.pool.map(_run_job, jobs, chunksize=max(1, len(jobs) // (self.workers * 4)))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.close()
            self.pool.join()

    def __enter__(self) -> BatchRunner:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def run_batch(seeds: list[int], params: dict, workers: int | None = None) -> list[EpisodeResult]:
    """Run the randomized-cube task for many seeds in parallel (used by the cube learning loop)."""
    jobs = [(s, params, "") for s in seeds]
    workers = workers or default_workers()
    if workers == 1 or len(seeds) < 4:
        _init_worker()
        return [_run_job(j) for j in jobs]
    with BatchRunner(workers) as runner:
        return runner.run(jobs)
