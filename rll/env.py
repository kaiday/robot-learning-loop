"""Franka Panda pick-and-place environment with per-seed domain randomization.

Supports several object shapes (box, cylinder, sphere, capsule lying down) and optional static
clutter blocks. Everything is drawn from a seeded RNG, so any episode replays bit-for-bit.
`sample_config(seed)` is the original randomized-cube task; `rll.objects` builds unfamiliar objects.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PANDA_SCENE = ROOT / "third_party" / "mujoco_menagerie" / "franka_emika_panda" / "scene.xml"

HOME_QPOS = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853, 0.04, 0.04])
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0
CONTROL_DT = 0.02  # 50 Hz controller; physics runs at the model timestep underneath
TARGET_TOL = 0.03  # metres, object centre to target centre for success
GRIPPER_KP = 800.0  # stiffer than Menagerie default (100) so grip force is ~15 N, like a real Panda hand
N_CLUTTER = 3
CLUTTER_HALF = (0.012, 0.012, 0.035)
SHAPES = ("box", "cylinder", "sphere", "capsule")
# Grasp yaw symmetry (radians) and offset per shape: a cube looks the same every 90 deg; a lying capsule
# every 180 deg and must be pinched across its axis (fingers close along world x at yaw 0, the bar lies along y); cylinders/spheres don't care (0 = any yaw).
YAW_SYMMETRY = {"box": (math.pi / 2, 0.0), "cylinder": (0.0, 0.0), "sphere": (0.0, 0.0), "capsule": (math.pi, 0.0)}


@dataclass(frozen=True)
class EpisodeConfig:
    seed: int
    obj_x: float
    obj_y: float
    obj_yaw: float
    half_size: float  # half-width the fingers close on
    mass: float
    friction: float
    obs_noise: float
    target_x: float
    target_y: float
    shape: str = "box"
    obj_name: str = "cube"
    size_b: float = 0.0  # cylinder half-height / capsule half-length
    clutter: tuple = field(default_factory=tuple)  # ((x, y), ...) static blocks
    physics: str = "default"  # "default" = MuJoCo soft contacts (objects creep under load); "realistic" = elliptic cone

    @property
    def half_height(self) -> float:
        """Height of the object's centre when resting on the floor."""
        return self.size_b if self.shape == "cylinder" else self.half_size

    def to_dict(self) -> dict:
        d = asdict(self)
        d["clutter"] = [list(c) for c in self.clutter]
        return d


def sample_config(seed: int) -> EpisodeConfig:
    """The original task: a randomized cube on a clean floor."""
    rng = np.random.default_rng(seed)
    return EpisodeConfig(
        seed=seed,
        obj_x=float(rng.uniform(0.42, 0.62)),
        obj_y=float(rng.uniform(-0.18, 0.05)),
        obj_yaw=float(rng.uniform(-math.pi / 4, math.pi / 4)),
        half_size=float(rng.uniform(0.016, 0.028)),
        mass=float(rng.uniform(0.05, 0.6)),
        friction=float(rng.uniform(0.35, 1.2)),
        obs_noise=float(rng.uniform(0.0, 0.006)),
        target_x=float(rng.uniform(0.40, 0.60)),
        target_y=float(rng.uniform(0.18, 0.30)),
    )


def _build_model(shape: str, physics: str = "default") -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
    for key in list(spec.keys):  # home key has no object dofs; we set qpos by hand
        spec.delete(key)

    gtype = {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    }[shape]
    obj = spec.worldbody.add_body(name="object", pos=[0.5, 0, 0.05])
    obj.add_freejoint(name="object_free")
    geom = obj.add_geom(
        name="object_geom",
        type=gtype,
        size=[0.02, 0.02, 0.02],
        rgba=[0.9, 0.35, 0.1, 1],
        priority=1,  # object friction wins over finger/floor friction, so randomization matters
        condim=4,
    )
    if shape == "capsule":  # lie it down: capsule axis along the body's y axis
        geom.quat = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]
    for i in range(N_CLUTTER):
        spec.worldbody.add_geom(
            name=f"clutter{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(CLUTTER_HALF),
            pos=[3 + i, 3, CLUTTER_HALF[2]], rgba=[0.45, 0.47, 0.5, 1],
        )
    spec.worldbody.add_site(
        name="target", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[TARGET_TOL, 0.001, 0],
        pos=[0.5, 0.25, 0.001], rgba=[0.1, 0.8, 0.3, 0.5],
    )
    spec.body("hand").add_site(name="tcp", pos=[0, 0, 0.1034], size=[0.005, 0, 0], rgba=[1, 0, 0, 0.5])
    model = spec.compile()
    if physics == "realistic":  # elliptic friction cone + high impratio: grasped objects stop creeping out
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        model.opt.impratio = 5.0

    grip = model.actuator("actuator8").id
    model.actuator_gainprm[grip, 0] = 0.04 * GRIPPER_KP / 255.0
    model.actuator_biasprm[grip, 1:3] = [-GRIPPER_KP, -0.1 * GRIPPER_KP]
    return model


def _geometry(cfg: EpisodeConfig) -> tuple[list[float], float, list[float], np.ndarray]:
    """geom_size, rbound, aabb half-extents (geom frame) and body inertia (body frame) for cfg's object."""
    r, b, m = cfg.half_size, cfg.size_b, cfg.mass
    if cfg.shape == "box":
        return [r, r, r], r * math.sqrt(3), [r, r, r], np.full(3, m * (2 * r) ** 2 / 6)
    if cfg.shape == "cylinder":
        t = m * (3 * r**2 + (2 * b) ** 2) / 12
        return [r, b, 0], math.hypot(r, b), [r, r, b], np.array([t, t, m * r**2 / 2])
    if cfg.shape == "sphere":
        return [r, 0, 0], r, [r, r, r], np.full(3, 0.4 * m * r**2)
    # capsule lying along body y: approximate inertia as a cylinder of the full length
    t = m * (3 * r**2 + (2 * (b + r)) ** 2) / 12
    return [r, b, 0], r + b, [r, r, b + r], np.array([t, m * r**2 / 2, t])


class PickPlaceEnv:
    def __init__(self, shape: str = "box", physics: str = "default") -> None:
        self._models: dict[tuple[str, str], tuple[mujoco.MjModel, mujoco.MjData]] = {}
        self.cfg: EpisodeConfig | None = None
        self.obs_rng = np.random.default_rng(0)
        self._use(shape, physics)

    def _use(self, shape: str, physics: str = "default") -> None:
        key = (shape, physics)
        if key not in self._models:
            model = _build_model(shape, physics)
            self._models[key] = (model, mujoco.MjData(model))
        self.model, self.data = self._models[key]
        m = self.model
        self.obj_body = m.body("object").id
        self.obj_geom = m.geom("object_geom").id
        self.obj_qadr = m.joint("object_free").qposadr[0]
        self.tcp_site = m.site("tcp").id
        self.target_site = m.site("target").id
        self.floor_geom = m.geom("floor").id
        self.clutter_geoms = [m.geom(f"clutter{i}").id for i in range(N_CLUTTER)]
        self.left_finger = m.body("left_finger").id
        self.right_finger = m.body("right_finger").id
        self.n_substeps = max(1, round(CONTROL_DT / m.opt.timestep))

    # -- episode setup ---------------------------------------------------------------
    def reset(self, cfg: EpisodeConfig) -> None:
        self._use(cfg.shape, cfg.physics)
        m, d = self.model, self.data
        self.cfg = cfg
        self.obs_rng = np.random.default_rng(cfg.seed + 7_919)
        size, rbound, aabb, inertia = _geometry(cfg)
        g = self.obj_geom
        m.geom_size[g] = size
        m.geom_rbound[g] = rbound
        m.geom_aabb[g] = [0, 0, 0, *aabb]
        m.geom_friction[g, 0] = cfg.friction
        m.body_mass[self.obj_body] = cfg.mass
        m.body_inertia[self.obj_body] = inertia
        m.site_pos[self.target_site] = [cfg.target_x, cfg.target_y, 0.001]
        for i, cg in enumerate(self.clutter_geoms):
            x, y = cfg.clutter[i] if i < len(cfg.clutter) else (3.0 + i, 3.0)
            m.geom_pos[cg] = [x, y, CLUTTER_HALF[2]]

        mujoco.mj_resetData(m, d)
        mujoco.mj_setConst(m, d)  # recompute mass-dependent constants (invweight0) or the solver goes unstable
        d.qpos[:9] = HOME_QPOS
        a = self.obj_qadr
        d.qpos[a : a + 3] = [cfg.obj_x, cfg.obj_y, cfg.half_height]
        d.qpos[a + 3 : a + 7] = [math.cos(cfg.obj_yaw / 2), 0, 0, math.sin(cfg.obj_yaw / 2)]
        d.ctrl[:7] = HOME_QPOS[:7]
        d.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_forward(m, d)

    def step(self, ctrl: np.ndarray) -> None:
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # -- observations ------------------------------------------------------------------
    def observe_object(self) -> tuple[np.ndarray, float]:
        """Noisy 'camera' estimate of object position and yaw (what the controller is allowed to see)."""
        assert self.cfg is not None
        pos = self.obj_pos + self.obs_rng.normal(0, self.cfg.obs_noise, 3) * [1, 1, 0]
        yaw = self.obj_yaw + self.obs_rng.normal(0, self.cfg.obs_noise * 5)
        return pos, yaw

    @property
    def obj_pos(self) -> np.ndarray:
        return self.data.xpos[self.obj_body].copy()

    @property
    def obj_yaw(self) -> float:
        q = self.data.xquat[self.obj_body]
        return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site].copy()

    @property
    def target_pos(self) -> np.ndarray:
        assert self.cfg is not None
        return np.array([self.cfg.target_x, self.cfg.target_y, self.cfg.half_height])

    @property
    def finger_width(self) -> float:
        return float(self.data.qpos[7] + self.data.qpos[8])

    def contacts(self) -> dict[str, bool]:
        d, m = self.data, self.model
        left = right = floor = arm = clutter = False
        clutter_ids = self.clutter_geoms
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if g1 in clutter_ids or g2 in clutter_ids:
                other = g2 if g1 in clutter_ids else g1
                if m.geom_bodyid[other] not in (0, self.obj_body):  # robot touched a clutter block
                    clutter = True
                continue
            if self.obj_geom not in (g1, g2):
                continue
            other = g2 if g1 == self.obj_geom else g1
            body = m.geom_bodyid[other]
            if body == self.left_finger:
                left = True
            elif body == self.right_finger:
                right = True
            elif other == self.floor_geom:
                floor = True
            elif body != self.obj_body:
                arm = True
        return {"left": left, "right": right, "floor": floor, "arm": arm, "clutter": clutter}
