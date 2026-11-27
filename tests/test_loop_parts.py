import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import rll.gate as gate_mod
from rll.brains import CodexBrain, HeuristicBrain
from rll.env import sample_config
from rll.runner import EpisodeResult
from rll.storage import WorkerStorage

def DRV(c):
    return {"condition": c, "failure_rate_by_tercile": {"low": 0.1, "mid": 0.2, "high": 0.6},
            "tercile_edges": [0, 1, 2, 3], "spread": 0.5}


REPORT = {
    "episodes": 10,
    "success_rate": 0.5,
    "failure_counts": {"slip": 4, "missed_grasp": 1},
    "failures": {
        "slip": {"count": 4, "rate": 0.4, "phases": {"lift": 4}, "drivers": [DRV("mass")], "examples": []},
        "missed_grasp": {"count": 1, "rate": 0.1, "phases": {"close": 1},
                         "drivers": [DRV("obj_yaw_abs")], "examples": []},
    },
}


def _res(seed, ok):
    return EpisodeResult(sample_config(seed), ok, None, 0.0)


def test_heuristic_targets_dominant_failure(v0_params, tmp_path):
    p = HeuristicBrain().propose({**REPORT, "params": v0_params}, v0_params, tmp_path)
    assert p.changes["lift_speed"] < v0_params["lift_speed"]
    assert "slip" in p.diagnosis


def test_gate_accepts_gain_and_rejects_regressions(monkeypatch):
    seeds = list(range(10))
    base = [_res(s, s < 5) for s in seeds]
    monkeypatch.setattr(gate_mod, "run_batch", lambda ss, p: [_res(s, s < 8) for s in ss])
    g, _ = gate_mod.run_gate(base, {}, seeds, [])
    assert g.accepted and g.fixed == 3 and g.regressions == 0

    monkeypatch.setattr(gate_mod, "run_batch", lambda ss, p: [_res(s, s >= 2) for s in ss])  # 80%, but breaks 0 and 1
    g, _ = gate_mod.run_gate(base, {}, seeds, [], max_regress_frac=0.0)
    assert not g.accepted and g.regressions == 2


def test_codex_brain_parses_structured_answer(monkeypatch, tmp_path, v0_params):
    seen = {}

    def fake_run(cmd, input, **kw):
        seen["cmd"], seen["prompt"] = cmd, input
        answer = {
            "diagnosis": "heavy cubes slip",
            "expected_effect": "+10%",
            "changes": [{"param": "lift_speed", "value": 0.2, "reason": "inertia"},
                        {"param": "align_yaw", "value": True, "reason": "corners"}],
        }
        with open(cmd[cmd.index("-o") + 1], "w") as f:
            f.write(json.dumps(answer))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    brain = CodexBrain()
    brain.remember(1, {"close_wait_s": 0.3}, 0.5, 0.45, False)
    p = brain.propose({**REPORT, "params": v0_params}, v0_params, tmp_path / "ws")
    assert p.changes == {"lift_speed": 0.2, "align_yaw": True}
    assert seen["cmd"][seen["cmd"].index("-s") + 1] == "read-only"
    assert "REJECTED" in seen["prompt"]
    assert (tmp_path / "ws" / "report.md").exists() and (tmp_path / "ws" / "controller.py").exists()
    assert p.apply(v0_params)["lift_speed"] == 0.2


def test_worker_storage_uploads(tmp_path):
    got = []

    class Handler(BaseHTTPRequestHandler):
        def _ok(self):
            n = int(self.headers["Content-Length"])
            got.append((self.command, self.path, self.headers["Authorization"], self.rfile.read(n)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        do_PUT = do_POST = _ok

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.json").write_text("{}")
    st = WorkerStorage(f"http://127.0.0.1:{srv.server_port}", "tok")
    assert st.upload_dir(tmp_path, "runs/r1") == 1
    st.record_iteration("r1", {"iteration": 1})
    srv.shutdown()
    assert got[0][:3] == ("PUT", "/blobs/runs/r1/a/x.json", "Bearer tok")
    assert got[1][:2] == ("POST", "/runs/r1/iterations")
