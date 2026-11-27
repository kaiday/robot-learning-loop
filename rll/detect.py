"""Find the failure: label *what* went wrong, *when*, and in *which phase*, from the captured log only.

Failure types, checked in causal order (the first one that fires is the root cause):
  hit_clutter   the robot touched a clutter block before releasing the object
  knocked       object pushed >1.5 cm before the gripper closed (bad approach / bad perception)
  missed_grasp  gripper finished closing without both fingers on the object
  slip          object slid out of the fingers during lift / transport / lower
  misplaced     sequence finished but object is off target or not resting on the floor
  timeout       controller never finished
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rll.env import TARGET_TOL

if TYPE_CHECKING:
    from rll.runner import EpisodeResult

KNOCK_TOL = 0.015
SLIP_TOL = 0.02
CARRY_PHASES = ("lift", "transport", "lower")
BEFORE_RELEASE = ("approach", "descend", "close", *CARRY_PHASES)


def _event(kind: str, row, detail: dict) -> dict:
    return {
        "type": kind,
        "step": int(row.step),
        "t": float(row.t),
        "phase": str(row.phase),
        "detail": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in detail.items()},
    }


def detect_failure(result: EpisodeResult) -> dict | None:
    if result.success:
        return None
    df = result.steps
    cfg = result.cfg
    if df.empty:
        return {"type": "timeout", "step": 0, "t": 0.0, "phase": "approach", "detail": {}}

    # hit_clutter: the arm or fingers touched a clutter block
    if "contact_clutter" in df:
        hits = df[df.contact_clutter & df.phase.isin(BEFORE_RELEASE)]
        if not hits.empty:
            row = hits.iloc[0]
            return _event("hit_clutter", row, {"finger_width": float(row.finger_width), "tcp_z": float(row.tcp_z),
                                               "n_clutter": len(cfg.clutter)})

    # knocked: object moved before the gripper closed
    pre = df[df.phase.isin(["approach", "descend"])]
    moved = np.hypot(pre.obj_x - cfg.obj_x, pre.obj_y - cfg.obj_y)
    hit = pre[moved > KNOCK_TOL]
    if not hit.empty:
        row = hit.iloc[0]
        return _event("knocked", row, {"moved_m": float(moved[hit.index[0]]), "arm_contact": bool(row.contact_arm)})

    # missed_grasp: at the end of close, both fingers must touch the object
    close = df[df.phase == "close"]
    if close.empty:
        return _event("timeout", df.iloc[-1], {})
    end = close.iloc[-1]
    if not (end.contact_left and end.contact_right):
        return _event(
            "missed_grasp",
            end,
            {"finger_width": float(end.finger_width), "left": bool(end.contact_left), "right": bool(end.contact_right)},
        )

    # slip: object drops away from the tcp while being carried
    carry = df[df.phase.isin(CARRY_PHASES)]
    if not carry.empty:
        offset0 = float(end.tcp_z - end.obj_z)
        gap = (carry.tcp_z - carry.obj_z) - offset0
        lost = carry[(gap > SLIP_TOL) | ~(carry.contact_left & carry.contact_right)]
        if not lost.empty:
            row = lost.iloc[0]
            vz = float(np.gradient(carry.tcp_z.to_numpy(), carry.t.to_numpy())[carry.index.get_loc(row.name)])
            return _event("slip", row, {"gap_m": float(gap[row.name]), "tcp_vz": vz, "tcp_z": float(row.tcp_z)})

    last = df.iloc[-1]
    if last.phase != "retreat":
        return _event("timeout", last, {})
    return _event(
        "misplaced",
        last,
        {"dist_m": result.final_dist, "obj_z": float(last.obj_z), "tol_m": TARGET_TOL,
         "resting": bool(last.obj_z < cfg.half_height + 0.01)},
    )
