import pandas as pd

from rll.analyze import build_report, report_markdown
from rll.detect import detect_failure
from rll.env import sample_config
from rll.runner import EpisodeResult, run_episode


def _episode(phases, seed=1):
    cfg = sample_config(seed)
    rows = [
        dict(step=i, t=0.02 * (i + 1), phase=ph, tcp_x=0.5, tcp_y=0.0, tcp_z=0.1, obj_x=cfg.obj_x,
             obj_y=cfg.obj_y, obj_z=0.02, obj_yaw=0.0, finger_width=0.04, gripper_cmd=0.0,
             contact_left=True, contact_right=True, contact_floor=False, contact_arm=False, contact_clutter=False)
        for i, ph in enumerate(phases)
    ]
    return cfg, pd.DataFrame(rows)


def test_success_has_no_failure():
    cfg, df = _episode(["approach"])
    assert detect_failure(EpisodeResult(cfg, True, df, 0.0)) is None


def test_missed_grasp_when_one_finger_off():
    cfg, df = _episode(["approach", "descend", "close", "close"])
    df.loc[3, "contact_right"] = False
    f = detect_failure(EpisodeResult(cfg, False, df, 0.2))
    assert f["type"] == "missed_grasp" and f["phase"] == "close" and f["step"] == 3


def test_knocked_when_cube_pushed_before_close():
    cfg, df = _episode(["approach", "descend", "close"])
    df.loc[1, "obj_x"] += 0.05
    assert detect_failure(EpisodeResult(cfg, False, df, 0.2))["type"] == "knocked"


def test_slip_when_cube_falls_during_lift():
    cfg, df = _episode(["close", "lift", "lift", "transport"])
    df.loc[2:, "tcp_z"] = 0.2  # tcp rises, cube stays on the table
    df.loc[2:, ["contact_left", "contact_right"]] = False
    f = detect_failure(EpisodeResult(cfg, False, df, 0.2))
    assert f["type"] == "slip" and f["phase"] == "lift" and f["step"] == 2


def test_misplaced_when_finished_off_target():
    cfg, df = _episode(["close", "lift", "release", "retreat"])
    assert detect_failure(EpisodeResult(cfg, False, df, 0.2))["type"] == "misplaced"


def test_report_on_real_v0_runs(env, v0_params):
    results = []
    for s in range(12):
        r = run_episode(env, s, v0_params)
        r.failure = detect_failure(r)
        results.append(r)
    rep = build_report(results, v0_params)
    assert rep["episodes"] == 12
    assert sum(rep["failure_counts"].values()) == sum(not r.success for r in results)
    for f in rep["failures"].values():
        assert f["drivers"] and f["examples"]
    assert "Failure report" in report_markdown(rep)
