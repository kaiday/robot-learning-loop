"""Scripted pick-and-place controller for the Panda.

Behaviour is driven entirely by policy/params.json, which is what the learning loop's
"brain" (heuristic or Codex) edits. Each phase is a named decision that the recorder logs.

Shape-aware: grasp height comes from the object height, grasp yaw from the shape symmetry.

Phases: approach -> descend -> close -> lift -> transport -> lower -> release -> retreat -> done
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from rll.env import GRIPPER_CLOSED, GRIPPER_OPEN, HOME_QPOS, YAW_SYMMETRY, PickPlaceEnv

PARAMS_PATH = Path(__file__).with_name("params.json")

# Bounds the loop enforces on any proposed params (the brain can't push values outside these).
PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "approach_height": (0.04, 0.25),
    "approach_speed": (0.05, 0.8),
    "descend_speed": (0.02, 0.5),
    "grasp_z_offset": (-0.02, 0.03),
    "close_wait_s": (0.05, 1.5),
    "lift_height": (0.05, 0.3),
    "lift_speed": (0.02, 0.8),
    "transport_speed": (0.05, 0.8),
    "release_height": (0.0, 0.1),
    "settle_s": (0.0, 1.0),
    "open_fraction": (0.3, 1.0),
    "grasp_yaw_offset": (-1.5708, 1.5708),  # extra hand rotation (rad), e.g. to open the fingers away from obstacles
}
# Values used when a params file predates a parameter.
PARAM_DEFAULTS = {"open_fraction": 1.0, "grasp_yaw_offset": 0.0}
BOOL_PARAMS = ("align_yaw", "remeasure_before_descend")


def load_params(path: Path = PARAMS_PATH) -> dict:
    return json.loads(Path(path).read_text())


def validate_params(params: dict) -> dict:
    """Clamp numbers into bounds and coerce booleans. Unknown keys are dropped."""
    out: dict = {}
    for k, (lo, hi) in PARAM_BOUNDS.items():
        out[k] = float(min(hi, max(lo, float(params.get(k, PARAM_DEFAULTS.get(k, lo))))))
    for k in BOOL_PARAMS:
        out[k] = bool(params.get(k, False))
    return out


@dataclass
class Segment:
    phase: str
    start: np.ndarray
    goal: np.ndarray
    duration: float
    gripper: float
    yaw: float
    hold: float = 0.0  # extra seconds to hold at the goal


@dataclass
class ScriptedPickPlace:
    env: PickPlaceEnv
    params: dict
    segments: list[Segment] = field(default_factory=list)
    _seg: int = 0
    _t: float = 0.0

    def __post_init__(self) -> None:
        self.params = validate_params(self.params)
        m = self.env.model
        self._ik = mujoco.MjData(m)
        self._q = HOME_QPOS[:7].copy()
        self._ik.qpos[:9] = HOME_QPOS
        mujoco.mj_kinematics(m, self._ik)
        self._r_home = self._ik.site_xmat[self.env.tcp_site].reshape(3, 3).copy()
        self._open = GRIPPER_OPEN * self.params["open_fraction"]
        self._lo = m.jnt_range[:7, 0]
        self._hi = m.jnt_range[:7, 1]
        self._plan_pick()

    # -- planning -----------------------------------------------------------------------
    def _grasp_yaw(self, observed_yaw: float) -> float:
        sym, offset = YAW_SYMMETRY[self.env.cfg.shape]
        extra = self.params["grasp_yaw_offset"]
        if not self.params["align_yaw"] or sym == 0:
            return extra
        # e.g. a cube looks the same every 90 degrees: pick the nearest face alignment.
        return (observed_yaw + offset + sym / 2) % sym - sym / 2 + extra

    def _plan_pick(self) -> None:
        p = self.params
        cube, yaw_obs = self.env.observe_object()
        h = self.env.cfg.half_height
        yaw = self._grasp_yaw(yaw_obs)
        start = self.env.tcp_pos
        above = np.array([cube[0], cube[1], h + p["approach_height"]])
        self.segments = [
            Segment("approach", start, above, _dur(start, above, p["approach_speed"]), self._open, yaw, hold=0.15),
        ]
        self._grasp_xy, self._yaw = cube[:2], yaw
        if not p["remeasure_before_descend"]:
            self._plan_descend_onwards()

    def _plan_descend_onwards(self) -> None:
        p, env = self.params, self.env
        h = env.cfg.half_height
        xy, yaw = self._grasp_xy, self._yaw
        above = np.array([xy[0], xy[1], h + p["approach_height"]])
        grasp = np.array([xy[0], xy[1], h + p["grasp_z_offset"]])
        lifted = np.array([xy[0], xy[1], h + p["lift_height"]])
        tgt = env.target_pos
        over_tgt = np.array([tgt[0], tgt[1], h + p["lift_height"]])
        place = np.array([tgt[0], tgt[1], h + p["release_height"] + 0.002])
        up = place + [0, 0, 0.1]
        s = p["transport_speed"]
        self.segments += [
            Segment("descend", above, grasp, _dur(above, grasp, p["descend_speed"]), self._open, yaw),
            Segment("close", grasp, grasp, p["close_wait_s"], GRIPPER_CLOSED, yaw),
            Segment("lift", grasp, lifted, _dur(grasp, lifted, p["lift_speed"]), GRIPPER_CLOSED, yaw),
            Segment("transport", lifted, over_tgt, _dur(lifted, over_tgt, s), GRIPPER_CLOSED, yaw),
            Segment("lower", over_tgt, place, _dur(over_tgt, place, p["descend_speed"]), GRIPPER_CLOSED, yaw),
            Segment("release", place, place, 0.3, GRIPPER_OPEN, yaw),
            Segment("retreat", place, up, _dur(place, up, s), GRIPPER_OPEN, yaw, hold=p["settle_s"]),
        ]

    # -- control ------------------------------------------------------------------------
    @property
    def phase(self) -> str:
        return self.segments[self._seg].phase if self._seg < len(self.segments) else "done"

    @property
    def done(self) -> bool:
        return self.phase == "done"

    def act(self, dt: float) -> np.ndarray:
        if self.done:
            return np.concatenate([self._q, [GRIPPER_OPEN]])
        seg = self.segments[self._seg]
        a = min(1.0, self._t / seg.duration) if seg.duration > 0 else 1.0
        a = 0.5 - 0.5 * math.cos(math.pi * a)  # smooth start/stop
        pos = seg.start + a * (seg.goal - seg.start)
        self._q = self._solve_ik(pos, seg.yaw)
        ctrl = np.concatenate([self._q, [seg.gripper]])
        self._t += dt
        if self._t >= seg.duration + seg.hold:
            self._advance()
        return ctrl

    def _advance(self) -> None:
        finished = self.segments[self._seg].phase
        self._seg += 1
        self._t = 0.0
        if finished == "approach" and self.params["remeasure_before_descend"]:
            # Second look from close range: wrist-camera estimate with a quarter of the noise.
            cube, yaw_obs = self.env.observe_object()
            true = self.env.obj_pos
            self._grasp_xy = true[:2] + (cube[:2] - true[:2]) * 0.25
            self._yaw = self._grasp_yaw(self.env.obj_yaw + (yaw_obs - self.env.obj_yaw) * 0.25)
            self._plan_descend_onwards()

    def _solve_ik(self, pos: np.ndarray, yaw: float, iters: int = 8) -> np.ndarray:
        m, d, site = self.env.model, self._ik, self.env.tcp_site
        c, s = math.cos(yaw), math.sin(yaw)
        r_des = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ self._r_home
        q = self._q.copy()
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            d.qpos[:7] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            r = d.site_xmat[site].reshape(3, 3)
            e_pos = pos - d.site_xpos[site]
            e_rot = 0.5 * sum(np.cross(r[:, i], r_des[:, i]) for i in range(3))
            err = np.concatenate([e_pos, e_rot])
            if np.linalg.norm(err) < 1e-4:
                break
            mujoco.mj_jacSite(m, d, jacp, jacr, site)
            j = np.vstack([jacp[:, :7], jacr[:, :7]])
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), err)
            null = (np.eye(7) - np.linalg.pinv(j) @ j) @ (HOME_QPOS[:7] - q) * 0.05
            q = np.clip(q + dq + null, self._lo, self._hi)
        return q


def _dur(a: np.ndarray, b: np.ndarray, speed: float) -> float:
    return max(0.2, float(np.linalg.norm(np.asarray(b) - np.asarray(a))) / speed)
