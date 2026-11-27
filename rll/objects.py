"""Unfamiliar objects and scenes.

Each object has a shape and size the robot can see, and a mass and friction it cannot see.
A *task* is an object in a scene ("clean" floor, or "row" = packed between two blocks, like on a shelf).
Every trial of a task re-randomizes the pose, camera noise, target and clutter placement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rll.env import EpisodeConfig


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    shape: str  # box | cylinder | sphere | capsule
    half_size: float  # half-width the fingers close on (cube half-side, radius)
    size_b: float = 0.0  # cylinder half-height / capsule half-length
    mass: float = 0.2  # hidden from the brain
    friction: float = 0.8  # hidden from the brain
    description: str = ""

    @property
    def height(self) -> float:
        return 2 * (self.size_b if self.shape == "cylinder" else self.half_size)

    @property
    def length(self) -> float:
        """Longest horizontal extent."""
        if self.shape == "capsule":
            return 2 * (self.size_b + self.half_size)
        if self.shape == "box":
            return 2 * self.half_size
        return 2 * self.half_size

    def features(self) -> dict:
        """What a camera could tell you about the object. Used for library lookup and shown to the brain."""
        return {"shape": self.shape, "width_m": round(2 * self.half_size, 4),
                "height_m": round(self.height, 4), "length_m": round(self.length, 4)}


CATALOGUE: dict[str, ObjectSpec] = {o.name: o for o in [
    ObjectSpec("cube", "box", 0.022, mass=0.2, friction=0.8, description="4.4 cm cube (the familiar object)"),
    ObjectSpec("ball", "sphere", 0.026, mass=0.15, friction=0.7, description="5.2 cm ball"),
    ObjectSpec("heavy_ball", "sphere", 0.03, mass=0.55, friction=0.45, description="6 cm heavy ball"),
    ObjectSpec("tall_can", "cylinder", 0.028, 0.055, mass=0.35, friction=0.55, description="5.6 cm wide, 11 cm tall can"),
    ObjectSpec("wide_can", "cylinder", 0.036, 0.045, mass=0.3, friction=0.6, description="7.2 cm wide can (gripper opens 8 cm)"),
    ObjectSpec("puck", "cylinder", 0.03, 0.011, mass=0.12, friction=0.7, description="6 cm wide, 2.2 cm tall puck"),
    ObjectSpec("bar", "capsule", 0.016, 0.05, mass=0.15, friction=0.7, description="13 cm long, 3.2 cm thick bar lying down"),
    ObjectSpec("slippery_box", "box", 0.024, mass=0.45, friction=0.3, description="4.8 cm heavy, slippery block"),
]}
SCENES = ("clean", "row")


@dataclass(frozen=True)
class Task:
    obj: ObjectSpec
    scene: str = "clean"

    @property
    def name(self) -> str:
        return f"{self.obj.name}@{self.scene}"

    def context(self) -> dict:
        """The description given to the brain: visible features only."""
        scene = {"clean": "object alone on an empty floor",
                 "row": "object packed in a row between two 2.4 cm wide, 7 cm tall blocks, roughly along the "
                        "gripper's finger axis, gaps of about 2-3.5 cm"}[self.scene]
        return {"task": self.name, "object": self.obj.features(), "scene": scene}


def task(spec: str) -> Task:
    """'ball' or 'ball@row'."""
    name, _, scene = spec.partition("@")
    return Task(CATALOGUE[name], scene or "clean")


def sample_trial(t: Task, seed: int) -> EpisodeConfig:
    rng = np.random.default_rng(seed)
    o = t.obj
    x, y = float(rng.uniform(0.45, 0.60)), float(rng.uniform(-0.15, 0.02))
    yaw = float(rng.uniform(-math.pi / 2, math.pi / 2))
    clutter: tuple = ()
    if t.scene == "row":
        blocks = []
        for side in (0.0, math.pi):  # either side of the object, along world x = the finger axis at yaw 0
            ang = side + rng.uniform(-0.35, 0.35)
            d = o.half_size + 0.012 + rng.uniform(0.018, 0.035)
            blocks.append((round(x + d * math.cos(ang), 4), round(y + d * math.sin(ang), 4)))
        clutter = tuple(blocks)
    return EpisodeConfig(
        seed=seed, obj_x=x, obj_y=y, obj_yaw=yaw, half_size=o.half_size, mass=o.mass, friction=o.friction,
        obs_noise=float(rng.uniform(0.0, 0.006)), target_x=float(rng.uniform(0.40, 0.60)),
        target_y=float(rng.uniform(0.18, 0.30)), shape=o.shape, obj_name=o.name, size_b=o.size_b, clutter=clutter,
        physics="realistic",
    )
