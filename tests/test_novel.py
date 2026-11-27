import json
import subprocess

import numpy as np

from policy.controller import validate_params
from rll.adapt import adapt, trial_seeds
from rll.brains import CodexBrain, HeuristicBrain, Proposal
from rll.env import YAW_SYMMETRY
from rll.library import Memory, SuccessLibrary, distance
from rll.objects import CATALOGUE, sample_trial, task
from rll.runner import BatchRunner, run_episode
from rll.visual import save_grid


def test_every_shape_runs_and_is_deterministic(env, good_params):
    for name in ("ball", "tall_can", "bar", "puck"):
        cfg = sample_trial(task(name), 3)
        a, b = run_episode(env, cfg, good_params), run_episode(env, cfg, good_params)
        assert a.steps.equals(b.steps)
        assert abs(a.steps.obj_z.iloc[0] - cfg.half_height) < 0.01  # starts resting on the floor
        assert np.all(np.diff(a.steps.t) > 0)


def test_row_scene_places_blocks_along_finger_axis():
    cfg = sample_trial(task("cube@row"), 11)
    assert len(cfg.clutter) == 2
    for x, y in cfg.clutter:
        assert abs(x - cfg.obj_x) > abs(y - cfg.obj_y)  # mostly along world x
    assert sample_trial(task("cube"), 11).clutter == ()


def test_grasp_yaw_respects_shape_symmetry(env, good_params):
    from policy.controller import ScriptedPickPlace

    for shape, name in (("sphere", "ball"), ("capsule", "bar")):
        cfg = sample_trial(task(name), 5)
        env.reset(cfg)
        ctl = ScriptedPickPlace(env, {**good_params, "grasp_yaw_offset": 0.3})
        sym, _ = YAW_SYMMETRY[shape]
        y = ctl._grasp_yaw(1.0)
        if sym == 0:
            assert y == 0.3
        else:
            assert abs(((y - 0.3) - 1.0 + sym / 2) % sym - sym / 2) < 1e-9


def test_old_params_files_still_load():
    p = validate_params({"approach_height": 0.1, "approach_speed": 0.4, "descend_speed": 0.15, "grasp_z_offset": 0,
                         "close_wait_s": 0.3, "lift_height": 0.15, "lift_speed": 0.2, "transport_speed": 0.3,
                         "release_height": 0.005, "settle_s": 0.3, "align_yaw": True, "remeasure_before_descend": True})
    assert p["open_fraction"] == 1.0 and p["grasp_yaw_offset"] == 0.0


def test_library_matches_similar_objects(tmp_path, v0_params):
    lib = SuccessLibrary(tmp_path / "lib.json")
    assert lib.lookup(CATALOGUE["ball"].features(), "row") is None
    lib.save(Memory("puck@row", CATALOGUE["puck"].features(), "row", v0_params, 1.0, 8, 1))
    hit = lib.lookup(CATALOGUE["ball"].features(), "row")
    assert hit and hit[0].task == "puck@row"  # both round
    assert lib.lookup(CATALOGUE["bar"].features(), "clean") is None  # different family and scene
    assert len(SuccessLibrary(tmp_path / "lib.json")) == 1  # persisted
    f = CATALOGUE["cube"].features()
    assert distance(f, f, "row", "row") == 0


class _ScriptedBrain:
    """Returns the same hypotheses every round; the second one is the real fix for puck@row."""

    def propose_many(self, report, params, workspace, images=None, k=4, attempts=None):
        assert "mass" not in json.dumps(report["failures"])  # hidden properties never reach the brain
        assert report["context"]["object"]["shape"] == "cylinder"
        return [Proposal({"lift_speed": 0.2}, "d", name="slower"),
                Proposal({"grasp_yaw_offset": 1.5708, "grasp_z_offset": 0.0}, "d", name="rotate")][:k]


def test_adapt_learns_saves_and_reuses(tmp_path, v0_params):
    start = json.loads(open("policy/params_cube_tuned.json").read())
    lib = SuccessLibrary(tmp_path / "lib.json")
    with BatchRunner(workers=4) as runner:
        r1 = adapt(task("puck@row"), _ScriptedBrain(), runner, start, tmp_path / "a", library=lib,
                   trials=4, k=2, max_rounds=2, threshold=0.75, log=lambda m: None)
        assert r1.first_try < 0.75 and r1.success and r1.rounds_used == 1
        assert r1.rounds[0].kept == "rotate" and len(lib) == 1
        assert (tmp_path / "a" / "round_1.png").exists()
        r2 = adapt(task("puck@row"), _ScriptedBrain(), runner, start, tmp_path / "b", library=lib,
                   trials=4, k=2, max_rounds=2, threshold=0.75, log=lambda m: None)
    assert r2.start_from == "puck@row" and r2.rounds_used == 0 and r2.first_try >= 0.75


def test_trial_seeds_are_stable_and_disjoint():
    t = task("ball@row")
    assert trial_seeds(t, 3) == trial_seeds(t, 3)
    assert not set(trial_seeds(t, 8)) & set(trial_seeds(t, 8, salt=1))


def test_heuristic_many_and_codex_many(monkeypatch, tmp_path, v0_params):
    report = {"episodes": 4, "success_rate": 0.0, "failure_counts": {"hit_clutter": 4},
              "failures": {"hit_clutter": {"count": 4, "rate": 1.0, "phases": {"descend": 4}, "drivers": [],
                                           "examples": []}}, "params": v0_params, "context": {"task": "x"}}
    hs = HeuristicBrain().propose_many(report, v0_params, tmp_path, k=4)
    assert len(hs) == 4 and len({h.name for h in hs}) == 4

    def fake_run(cmd, input, **kw):
        assert "exactly 3 DIFFERENT" in input and "round 1: a" in input
        ans = {"diagnosis": "fingers hit the blocks", "hypotheses": [
            {"name": "rotate", "rationale": "r", "changes": [{"param": "grasp_yaw_offset", "value": 1.57, "reason": "x"}]},
            {"name": "narrow", "rationale": "r", "changes": [{"param": "open_fraction", "value": 0.5, "reason": "x"}]},
            {"name": "both", "rationale": "r", "changes": []}]}
        with open(cmd[cmd.index("-o") + 1], "w") as f:
            f.write(json.dumps(ans))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = CodexBrain().propose_many(report, v0_params, tmp_path / "ws", k=3,
                                    attempts=[{"round": 1, "name": "a", "changes": {}, "success": 0.1}])
    assert [h.name for h in out] == ["rotate", "narrow", "both"]
    assert out[1].apply(v0_params)["open_fraction"] == 0.5


def test_grid_image(tmp_path):
    f = np.zeros((352, 480, 3), np.uint8)
    p = save_grid([(f, True, "a: 100%"), (f, False, "b: 0%")], tmp_path / "g.png", cols=2, title="t")
    assert p.exists() and p.stat().st_size > 0
