"""The 'Fix' step. A brain reads the failure report and proposes parameter changes.

- HeuristicBrain: hand-written rules, offline, deterministic (tests + fallback).
- CodexBrain: `codex exec` reads the report, the controller source and failure keyframes,
  and returns a structured proposal. It runs read-only: it proposes, the gate decides.

`propose` returns one fix (cube learning loop). `propose_many` returns K different hypotheses that
are tested in parallel (unfamiliar-object adaptation).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from policy.controller import BOOL_PARAMS, PARAM_BOUNDS, validate_params
from rll.analyze import report_markdown

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Proposal:
    changes: dict
    diagnosis: str
    expected_effect: str = ""
    brain: str = ""
    raw: dict = field(default_factory=dict)
    seconds: float = 0.0
    name: str = ""

    def apply(self, params: dict) -> dict:
        return validate_params({**params, **self.changes})

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "brain": self.brain,
            "diagnosis": self.diagnosis,
            "changes": self.changes,
            "expected_effect": self.expected_effect,
            "seconds": round(self.seconds, 1),
        }


class HeuristicBrain:
    name = "heuristic"

    def propose(self, report: dict, params: dict, workspace: Path, images: list[Path] | None = None) -> Proposal:
        fails = report["failures"]
        if not fails:
            return Proposal({}, "No failures to fix.", brain=self.name)
        kind = max(fails, key=lambda k: fails[k]["count"])
        drivers = [d["condition"] for d in fails[kind]["drivers"]]
        p, ch = validate_params(params), {}
        if kind == "slip":
            if p["grasp_z_offset"] > 0.004:
                ch["grasp_z_offset"] = round(p["grasp_z_offset"] - 0.006, 4)
            ch["lift_speed"] = round(p["lift_speed"] * 0.6, 3)
            ch["transport_speed"] = round(p["transport_speed"] * 0.7, 3)
            ch["close_wait_s"] = round(p["close_wait_s"] + 0.15, 3)
        elif kind == "missed_grasp":
            ch["grasp_z_offset"] = round(p["grasp_z_offset"] - 0.005, 4)
            ch["close_wait_s"] = round(p["close_wait_s"] + 0.2, 3)
            if "obj_yaw_abs" in drivers:
                ch["align_yaw"] = True
            if "obs_noise" in drivers:
                ch["remeasure_before_descend"] = True
        elif kind == "knocked":
            ch["remeasure_before_descend"] = True
            ch["align_yaw"] = True
            ch["descend_speed"] = round(p["descend_speed"] * 0.7, 3)
        elif kind == "hit_clutter":
            ch["open_fraction"] = round(max(0.3, p["open_fraction"] - 0.2), 3)
        elif kind == "misplaced":
            ch["release_height"] = round(p["release_height"] * 0.5, 4)
            ch["settle_s"] = round(p["settle_s"] + 0.2, 3)
        else:
            ch["approach_speed"] = round(p["approach_speed"] * 1.2, 3)
        return Proposal(ch, f"Dominant failure '{kind}', drivers {drivers}.", brain=self.name, name=f"fix_{kind}")

    def propose_many(self, report: dict, params: dict, workspace: Path, images: list[Path] | None = None,
                     k: int = 4, attempts: list[dict] | None = None) -> list[Proposal]:
        """The rule-based fix plus simple variations around it."""
        p = validate_params(params)
        base = self.propose(report, p, workspace)
        variants = [
            base,
            Proposal({**base.changes, "grasp_z_offset": 0.0}, base.diagnosis + " + grip at centre height",
                     brain=self.name, name="centre_grip"),
            Proposal({**base.changes, "lift_speed": round(p["lift_speed"] * 0.5, 3), "close_wait_s": 0.6},
                     base.diagnosis + " + slow lift", brain=self.name, name="slow_lift"),
            Proposal({**base.changes, "grasp_yaw_offset": 1.5708 if abs(p["grasp_yaw_offset"]) < 0.1 else 0.0},
                     base.diagnosis + " + rotate grip 90 deg", brain=self.name, name="rotate_grip"),
            Proposal({**base.changes, "open_fraction": round(max(0.3, p["open_fraction"] - 0.3), 3)},
                     base.diagnosis + " + narrower opening", brain=self.name, name="narrow_open"),
        ]
        return variants[:k]


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["diagnosis", "changes", "expected_effect"],
    "properties": {
        "diagnosis": {"type": "string", "description": "Root cause of the dominant failures, citing the evidence."},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["param", "value", "reason"],
                "properties": {
                    "param": {"type": "string", "enum": [*PARAM_BOUNDS, *BOOL_PARAMS]},
                    "value": {"anyOf": [{"type": "number"}, {"type": "boolean"}]},
                    "reason": {"type": "string"},
                },
            },
        },
        "expected_effect": {"type": "string", "description": "Predicted change in success rate and failure mix."},
    },
}

MULTI_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["diagnosis", "hypotheses"],
    "properties": {
        "diagnosis": {"type": "string", "description": "What is going wrong and why, citing the evidence."},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "changes", "rationale"],
                "properties": {
                    "name": {"type": "string", "description": "short snake_case label"},
                    "changes": SCHEMA["properties"]["changes"],
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}

PROMPT = """You are the 'brain' in a robot learning loop: run -> capture -> find failure -> FIX -> run again.

A simulated Franka Panda picks a randomized cube and places it on a target. Its behaviour is a scripted
controller (controller.py, read it) driven entirely by params.json. Every episode was recorded, failures were
detected automatically, and report.md summarises them: failure types, the phase they happen in, which
randomized conditions drive them (failure rate per tercile), and example events. Images attached are
keyframes around real failures (before / at / after the failure step).

Your job: propose a parameter change that will raise the success rate on NEW random episodes.
- Diagnose root causes from the evidence; explain the physics (e.g. inertia, grip force, contact point).
- Change only the params that address the diagnosis. Prefer a few well-reasoned changes over many.
- Allowed params and bounds: {bounds}. Booleans: {bools}.
- Episodes time out at 15 s, so don't make everything extremely slow.
- A regression gate will replay held-out seeds; changes that make things worse are rejected.
{history}
Do not modify any files. Reply only with the JSON described by the output schema."""

MULTI_PROMPT = """You are the 'brain' of a robot that is meeting an UNFAMILIAR object and must learn to pick it up
by retrying: attempt -> capture -> find failure -> FIX -> attempt again.

A simulated Franka Panda (parallel gripper, max opening 8 cm, fingers close along the hand's local x axis,
which is world x at grasp yaw 0) must pick the object and place it on a target. Its behaviour is a
scripted controller (controller.py, read it) driven entirely by params.json. report.md has the object and
scene description (what a camera can see; mass and friction are unknown), the current params, and an
automatic analysis of the last attempts: failure types, phases, drivers and example events. Images, if
attached, are keyframes before / at / after real failures.

Propose exactly {k} DIFFERENT hypotheses. All of them will be tested at the same time in parallel
simulations on the same trials, and the best one is kept. Make them genuinely different bets (different
root causes or different fixes), not small variations of one idea.
- Explain the physics/geometry behind each (contact point vs object shape, grip force vs weight and
  acceleration, finger clearance vs obstacles, perception error vs gripper opening).
- Each hypothesis lists only the params it changes, relative to the current params.
- Allowed params and bounds: {bounds}. Booleans: {bools}.
- Episodes time out at 15 s.
{history}
Do not modify any files. Reply only with the JSON described by the output schema."""


class CodexBrain:
    name = "codex"

    def __init__(self, model: str | None = None, timeout_s: int = 600, images: bool = True) -> None:
        self.model = model
        self.timeout_s = timeout_s
        self.use_images = images
        self.history: list[dict] = []  # what was tried and whether the gate accepted it

    def _prepare(self, ws: Path, report: dict, params: dict, schema: dict) -> Path:
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "report.md").write_text(report_markdown(report), encoding="utf-8")
        (ws / "report.json").write_text(json.dumps(report, indent=2))
        (ws / "params.json").write_text(json.dumps(params, indent=2))
        shutil.copy(ROOT / "policy" / "controller.py", ws / "controller.py")
        path = ws / "schema.json"
        path.write_text(json.dumps(schema))
        return path

    def _exec(self, ws: Path, prompt: str, schema: Path, images: list[Path] | None) -> tuple[dict, float]:
        out = ws / "codex_answer.json"
        cmd = [shutil.which("codex") or "codex", "exec"]
        for img in (images or []) if self.use_images else []:
            cmd += ["-i", str(img)]
        cmd += ["--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-C", str(ws),
                "--output-schema", str(schema), "-o", str(out), "--color", "never"]
        if self.model:
            cmd += ["-m", self.model]
        cmd.append("-")  # prompt from stdin
        t0 = time.time()
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                              timeout=self.timeout_s)
        (ws / "codex_stdout.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"codex exec failed ({proc.returncode}); see {ws / 'codex_stdout.log'}")
        return json.loads(out.read_text(encoding="utf-8")), time.time() - t0

    def propose(self, report: dict, params: dict, workspace: Path, images: list[Path] | None = None) -> Proposal:
        ws = Path(workspace)
        schema = self._prepare(ws, report, params, SCHEMA)
        history = ""
        if self.history:
            history = "\nPrevious attempts in this loop (learn from them):\n" + "\n".join(
                f"- iteration {h['iteration']}: changes {h['changes']} -> held-out success "
                f"{h['before']:.0%} -> {h['after']:.0%}, {'ACCEPTED' if h['accepted'] else 'REJECTED'}"
                for h in self.history
            ) + "\n"
        prompt = PROMPT.format(bounds=PARAM_BOUNDS, bools=list(BOOL_PARAMS), history=history)
        answer, secs = self._exec(ws, prompt, schema, images)
        return Proposal(
            changes={c["param"]: c["value"] for c in answer["changes"]},
            diagnosis=answer["diagnosis"],
            expected_effect=answer.get("expected_effect", ""),
            brain=self.name,
            raw=answer,
            seconds=secs,
        )

    def propose_many(self, report: dict, params: dict, workspace: Path, images: list[Path] | None = None,
                     k: int = 4, attempts: list[dict] | None = None) -> list[Proposal]:
        ws = Path(workspace)
        schema = self._prepare(ws, report, params, MULTI_SCHEMA)
        history = ""
        if attempts:
            history = "\nHypotheses already tested on this object (success rate on the same trials):\n" + "\n".join(
                f"- round {a['round']}: {a['name']} {a['changes']} -> {a['success']:.0%}" for a in attempts
            ) + "\n"
        prompt = MULTI_PROMPT.format(k=k, bounds=PARAM_BOUNDS, bools=list(BOOL_PARAMS), history=history)
        answer, secs = self._exec(ws, prompt, schema, images)
        out = []
        for h in answer["hypotheses"][:k]:
            out.append(Proposal(
                changes={c["param"]: c["value"] for c in h["changes"]},
                diagnosis=answer["diagnosis"], expected_effect=h["rationale"], brain=self.name,
                raw=h, seconds=secs, name=h["name"],
            ))
        return out

    def remember(self, iteration: int, changes: dict, before: float, after: float, accepted: bool) -> None:
        self.history.append(
            {"iteration": iteration, "changes": changes, "before": before, "after": after, "accepted": accepted}
        )
