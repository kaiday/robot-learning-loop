import json
from pathlib import Path

import pytest

from rll.env import PickPlaceEnv

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def env():
    return PickPlaceEnv()


@pytest.fixture
def v0_params():
    return json.loads((ROOT / "policy" / "params_v0.json").read_text())


@pytest.fixture
def good_params(v0_params):
    return {**v0_params, "grasp_z_offset": 0.0, "close_wait_s": 0.5, "lift_speed": 0.15, "transport_speed": 0.3,
            "align_yaw": True, "remeasure_before_descend": True}
