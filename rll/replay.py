"""Deterministic replay: re-run a failed seed with rendering on, save video + keyframes around the failure."""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from rll.detect import detect_failure
from rll.env import PickPlaceEnv
from rll.runner import EpisodeResult, run_episode, save_episode

RENDER_EVERY = 2  # control steps per frame (25 fps)


def replay_failure(env: PickPlaceEnv, seed: int, params: dict, out_dir: Path) -> tuple[EpisodeResult, Path | None]:
    with mujoco.Renderer(env.model, 360, 480) as renderer:
        res = run_episode(env, seed, params, render_every=RENDER_EVERY, renderer=renderer)
    res.failure = detect_failure(res)
    ep_dir = save_episode(res, out_dir, params)
    if not res.failure or not res.frames:
        return res, None
    i = res.failure["step"] // RENDER_EVERY
    picks = [max(0, i - 12), i, min(len(res.frames) - 1, i + 12)]  # ~0.5 s before / at / after
    strip = np.concatenate([res.frames[j] for j in picks], axis=1)
    png = ep_dir / "keyframes.png"
    iio.imwrite(png, strip)
    return res, png
