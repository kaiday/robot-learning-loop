"""Run episodes and capture everything the robot saw and did, step by step."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from policy.controller import ScriptedPickPlace
from rll.env import CONTROL_DT, TARGET_TOL, EpisodeConfig, PickPlaceEnv, sample_config

MAX_TIME_S = 15.0


@dataclass
class EpisodeResult:
    cfg: EpisodeConfig
    success: bool
    steps: pd.DataFrame
    final_dist: float
    failure: dict | None = None  # filled in by rll.detect
    frames: list[np.ndarray] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "seed": self.cfg.seed,
            "success": self.success,
            "final_dist": round(self.final_dist, 4),
            "failure_type": self.failure["type"] if self.failure else None,
            "failure_phase": self.failure["phase"] if self.failure else None,
            "failure_t": self.failure["t"] if self.failure else None,
            **{f"cfg_{k}": v for k, v in self.cfg.to_dict().items() if k != "seed"},
        }


def run_episode(
    env: PickPlaceEnv,
    seed: int,
    params: dict,
    render_every: int = 0,
    renderer: mujoco.Renderer | None = None,
) -> EpisodeResult:
    cfg = sample_config(seed)
    env.reset(cfg)
    ctl = ScriptedPickPlace(env, params)
    rows, frames = [], []
    cam = _camera(cfg) if renderer else None
    step = 0
    while env.data.time < MAX_TIME_S:
        phase = ctl.phase
        if ctl.done:
            break
        ctrl = ctl.act(CONTROL_DT)
        env.step(ctrl)
        c = env.contacts()
        cube, tcp = env.cube_pos, env.tcp_pos
        rows.append(
            {
                "step": step,
                "t": round(env.data.time, 4),
                "phase": phase,
                "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
                "cube_x": cube[0], "cube_y": cube[1], "cube_z": cube[2],
                "cube_yaw": env.cube_yaw,
                "finger_width": env.finger_width,
                "gripper_cmd": float(ctrl[7]),
                "contact_left": c["left"], "contact_right": c["right"],
                "contact_floor": c["floor"], "contact_arm": c["arm"],
            }
        )
        if renderer is not None and render_every and step % render_every == 0:
            renderer.update_scene(env.data, camera=cam)
            frames.append(renderer.render().copy())
        step += 1

    final = env.cube_pos
    dist = float(np.linalg.norm(final[:2] - env.target_pos[:2]))
    resting = final[2] < cfg.half_size + 0.01
    success = bool(ctl.done and dist < TARGET_TOL and resting)
    return EpisodeResult(cfg=cfg, success=success, steps=pd.DataFrame(rows), final_dist=dist, frames=frames)


def _camera(cfg: EpisodeConfig) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [(cfg.cube_x + cfg.target_x) / 2, (cfg.cube_y + cfg.target_y) / 2, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 0.9, 150, -30
    return cam


def save_episode(result: EpisodeResult, out_dir: Path, params: dict) -> Path:
    """Write steps.parquet, meta.json and (if rendered) video.mp4 for one episode."""
    ep = Path(out_dir) / f"ep_{result.cfg.seed:06d}"
    ep.mkdir(parents=True, exist_ok=True)
    result.steps.to_parquet(ep / "steps.parquet", index=False)
    meta = {**result.summary(), "config": result.cfg.to_dict(), "params": params, "failure": result.failure}
    (ep / "meta.json").write_text(json.dumps(meta, indent=2))
    if result.frames:
        import imageio.v3 as iio

        iio.imwrite(ep / "video.mp4", np.stack(result.frames), fps=int(1 / CONTROL_DT / 2), codec="libx264")
    return ep


# -- parallel batches -------------------------------------------------------------------
_worker_env: PickPlaceEnv | None = None


def _init_worker() -> None:
    global _worker_env
    _worker_env = PickPlaceEnv()


def _run_one(args: tuple[int, dict]) -> EpisodeResult:
    from rll.detect import detect_failure

    seed, params = args
    assert _worker_env is not None
    res = run_episode(_worker_env, seed, params)
    res.failure = detect_failure(res)
    return res


def run_batch(seeds: list[int], params: dict, workers: int | None = None) -> list[EpisodeResult]:
    """Run many seeds across processes. Results come back in seed order, with failures labelled."""
    import multiprocessing as mp
    import os

    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    if workers == 1 or len(seeds) < 4:
        _init_worker()
        return [_run_one((s, params)) for s in seeds]
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        return pool.map(_run_one, [(s, params) for s in seeds], chunksize=max(1, len(seeds) // (workers * 4)))
