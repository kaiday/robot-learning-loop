import numpy as np

from policy.controller import PARAM_BOUNDS, validate_params
from rll.env import sample_config
from rll.runner import run_episode


def test_config_is_seeded():
    assert sample_config(42) == sample_config(42)
    assert sample_config(42) != sample_config(43)


def test_replay_is_bit_exact(env, v0_params):
    a = run_episode(env, 123, v0_params).steps
    b = run_episode(env, 123, v0_params).steps
    assert a.equals(b)


def test_good_params_pick_and_place(env, good_params):
    results = [run_episode(env, s, good_params) for s in range(5)]
    assert sum(r.success for r in results) >= 4
    phases = list(dict.fromkeys(results[0].steps.phase))
    assert phases == ["approach", "descend", "close", "lift", "transport", "lower", "release", "retreat"]
    assert np.all(np.diff(results[0].steps.t) > 0)  # no solver resets


def test_validate_params_clamps_and_drops(v0_params):
    p = validate_params({**v0_params, "lift_speed": 99, "grasp_z_offset": -5, "align_yaw": 1, "junk": 3})
    assert p["lift_speed"] == PARAM_BOUNDS["lift_speed"][1]
    assert p["grasp_z_offset"] == PARAM_BOUNDS["grasp_z_offset"][0]
    assert p["align_yaw"] is True and "junk" not in p
