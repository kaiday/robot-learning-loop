"""Render the Menagerie robots to PNG, and optionally open the interactive MuJoCo viewer.

    uv run python scripts/show_robots.py              # write PNGs to out/robots/
    uv run python scripts/show_robots.py --view panda # open interactive window
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE = ROOT / "third_party" / "mujoco_menagerie"

ROBOTS = {
    # name: (scene file, camera lookat, distance, azimuth, elevation)
    "panda": ("franka_emika_panda/scene.xml", (0.3, 0.0, 0.45), 2.0, 135, -20),
    "so_arm100": ("trs_so_arm100/scene.xml", (0.0, -0.15, 0.15), 0.9, 135, -25),
    "go2": ("unitree_go2/scene.xml", (0.0, 0.0, 0.25), 1.6, 135, -20),
}


def load(name: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(MENAGERIE / ROBOTS[name][0]))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def render(name: str, width: int = 640, height: int = 480) -> np.ndarray:
    model, data = load(name)
    _, lookat, dist, az, el = ROBOTS[name]
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    with mujoco.Renderer(model, height, width) as r:
        r.update_scene(data, camera=cam)
        return r.render()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=ROBOTS, help="open the interactive viewer for one robot")
    args = ap.parse_args()

    if args.view:
        import mujoco.viewer

        model, data = load(args.view)
        mujoco.viewer.launch(model, data)
        return

    out = ROOT / "out" / "robots"
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for name in ROBOTS:
        img = render(name)
        iio.imwrite(out / f"{name}.png", img)
        frames.append(img)
        print(f"wrote {out / f'{name}.png'}")
    iio.imwrite(out / "all.png", np.concatenate(frames, axis=1))
    print(f"wrote {out / 'all.png'}")


if __name__ == "__main__":
    main()
