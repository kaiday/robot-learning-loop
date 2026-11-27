"""Deterministic replay: re-run a seed with rendering on, save video and/or keyframes around the failure."""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from rll.detect import detect_failure
from rll.env import EpisodeConfig, PickPlaceEnv
from rll.runner import EpisodeResult, run_episode, save_episode

RENDER_EVERY = 2  # control steps per video frame (25 fps)
KEY_OFFSET = 25  # control steps (~0.5 s) before/after the failure for keyframes


def keyframe_steps(failure_step: int, n_steps: int) -> list[int]:
    return [max(0, failure_step - KEY_OFFSET), failure_step, min(n_steps - 1, failure_step + KEY_OFFSET)]


def replay_keyframes(env: PickPlaceEnv, cfg: EpisodeConfig | int, params: dict, at_step: int | None = None,
                     n_steps: int | None = None) -> tuple[EpisodeResult, np.ndarray | None]:
    """Fast: render only 3 frames (before / at / after `at_step`, or the failure if not given)."""
    if at_step is None or n_steps is None:
        probe = run_episode(env, cfg, params)
        probe.failure = detect_failure(probe)
        at_step = probe.failure["step"] if probe.failure else len(probe.steps) - 1
        n_steps = len(probe.steps)
    res = run_episode(env, cfg, params, render_steps=set(keyframe_steps(at_step, n_steps)))
    res.failure = detect_failure(res)
    return res, (np.concatenate(res.frames, axis=1) if res.frames else None)


def replay_failure(env: PickPlaceEnv, cfg: EpisodeConfig | int, params: dict, out_dir: Path) -> tuple[EpisodeResult, Path | None]:
    """Full replay with video, saved to disk with keyframes.png."""
    res = run_episode(env, cfg, params, render_every=RENDER_EVERY)
    res.failure = detect_failure(res)
    ep_dir = save_episode(res, out_dir, params)
    if not res.failure or not res.frames:
        return res, None
    i = res.failure["step"] // RENDER_EVERY
    picks = [max(0, i - 12), i, min(len(res.frames) - 1, i + 12)]  # ~0.5 s before / at / after
    png = ep_dir / "keyframes.png"
    iio.imwrite(png, np.concatenate([res.frames[j] for j in picks], axis=1))
    return res, png
