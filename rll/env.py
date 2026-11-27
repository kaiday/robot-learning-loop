"""Franka Panda pick-and-place environment with per-seed domain randomization.

The cube's pose, size, mass, friction and the perception noise are drawn from a seeded RNG,
so any episode can be replayed bit-for-bit from its seed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_SCENE = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"

HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0
CONTROL_DT = 0.02  # 50 Hz controller; physics runs at the model timestep underneath
TARGET_TOL = 0.03  # metres, cube centre to target centre for success
GRIPPER_KP = 800.0  # stiffer than Menagerie default (100) so grip force is ~15 N, like a real Panda hand


@dataclass(frozen=True)
class EpisodeConfig:
    seed: int
    cube_x: float
    cube_y: float
    cube_yaw: float
    half_size: float
    mass: float
    friction: float
    obs_noise: float
    target_x: float
    target_y: float

    def to_dict(self) -> dict:
        return asdict(self)


def sample_config(seed: int) -> EpisodeConfig:
    rng = np.random.default_rng(seed)
    return EpisodeConfig(
        seed=seed,
        cube_x=float(rng.uniform(0.42, 0.62)),
        cube_y=float(rng.uniform(-0.18, 0.05)),
        cube_yaw=float(rng.uniform(-math.pi / 4, math.pi / 4)),
        half_size=float(rng.uniform(0.016, 0.028)),
        mass=float(rng.uniform(0.05, 0.6)),
        friction=float(rng.uniform(0.35, 1.2)),
        obs_noise=float(rng.uniform(0.0, 0.006)),
        target_x=float(rng.uniform(0.40, 0.60)),
        target_y=float(rng.uniform(0.18, 0.30)),
    )


def _build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
    for key in list(spec.keys):  # home key has no cube dofs; we set qpos by hand
        spec.delete(key)

    cube = spec.worldbody.add_body(name="cube", pos=[0.5, 0, 0.02])
    cube.add_freejoint(name="cube_free")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        rgba=[0.9, 0.35, 0.1, 1],
        priority=1,  # cube friction wins over finger/floor friction, so randomization matters
        condim=4,
    )
    spec.worldbody.add_site(
        name="target",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[TARGET_TOL, 0.001, 0],
        pos=[0.5, 0.25, 0.001],
        rgba=[0.1, 0.8, 0.3, 0.5],
    )
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005, 0, 0], rgba=[1, 0, 0, 0.5])
    model = spec.compile()

    grip = model.actuator("actuator8").id
    model.actuator_gainprm[grip, 0] = 0.04 * GRIPPER_KP / 255.0
    model.actuator_biasprm[grip, 1:3] = [-GRIPPER_KP, -0.1 * GRIPPER_KP]
    return model


class PickPlaceEnv:
    def __init__(self) -> None:
        self.model = _build_model()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.cube_body = m.body("cube").id
        self.cube_geom = m.geom("cube_geom").id
        self.cube_qadr = m.joint("cube_free").qposadr[0]
        self.tcp_site = m.site("tcp").id
        self.target_site = m.site("target").id
        self.floor_geom = m.geom("floor").id
        self.left_finger = m.body("left_finger").id
        self.right_finger = m.body("right_finger").id
        self.n_substeps = max(1, round(CONTROL_DT / m.opt.timestep))
        self.cfg: EpisodeConfig | None = None
        self.obs_rng = np.random.default_rng(0)

    # -- episode setup ---------------------------------------------------------------
    def reset(self, cfg: EpisodeConfig) -> None:
        m, d = self.model, self.data
        self.cfg = cfg
        self.obs_rng = np.random.default_rng(cfg.seed + 7_919)
        h = cfg.half_size
        m.geom_size[self.cube_geom] = [h, h, h]
        m.geom_rbound[self.cube_geom] = h * math.sqrt(3)
        m.geom_friction[self.cube_geom, 0] = cfg.friction
        m.body_mass[self.cube_body] = cfg.mass
        m.body_inertia[self.cube_body] = cfg.mass * (2 * h) ** 2 / 6
        m.site_pos[self.target_site] = [cfg.target_x, cfg.target_y, 0.001]

        mujoco.mj_resetData(m, d)
        mujoco.mj_setConst(m, d)  # recompute mass-dependent constants (invweight0) or the solver goes unstable
        d.qpos[:9] = HOME_QPOS
        a = self.cube_qadr
        d.qpos[a : a + 3] = [cfg.cube_x, cfg.cube_y, h]
        d.qpos[a + 3 : a + 7] = [math.cos(cfg.cube_yaw / 2), 0, 0, math.sin(cfg.cube_yaw / 2)]
        d.ctrl[:7] = HOME_QPOS[:7]
        d.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_forward(m, d)

    def step(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # -- observations ------------------------------------------------------------------
    def observe_cube(self) -> tuple[np.ndarray, float]:
        """Noisy 'camera' estimate of cube position and yaw (what the controller is allowed to see)."""
        assert self.cfg is not None
        pos = self.cube_pos + self.obs_rng.normal(0, self.cfg.obs_noise, 3) * [1, 1, 0]
        yaw = self.cube_yaw + self.obs_rng.normal(0, self.cfg.obs_noise * 5)
        return pos, yaw

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    @property
    def cube_yaw(self) -> float:
        q = self.data.xquat[self.cube_body]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    @property
    def target_pos(self) -> np.ndarray:
        assert self.cfg is not None
        return np.array([self.cfg.target_x, self.cfg.target_y, self.cfg.half_size])

    @property
    def finger_width(self) -> float:
        return float(self.data.qpos[7] + self.data.qpos[8])

    def contacts(self) -> dict[str, bool]:
        d, m = self.data, self.model
        left = right = floor = arm = False
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if self.cube_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.cube_geom else g1
            body = m.geom_bodyid[other]
            if body == self.left_finger:
                left = True
            elif body == self.right_finger:
                right = True
            elif other == self.floor_geom:
                floor = True
            elif body != self.cube_body:
                arm = True
        return {"left": left, "right": right, "floor": floor, "arm": arm}
