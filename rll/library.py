"""Success library: remember what worked, keyed by what the robot can see.

Each entry stores the task, the object's visible features, the params that made it reliable, how reliable
they were, and where the recording of a successful run lives. For a new object, `lookup` returns the
closest previous success as the starting point, or None for a cold start.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "out" / "library" / "library.json"
MATCH_THRESHOLD = 1.5  # feature distance above which a memory is too different to reuse


@dataclass
class Memory:
    task: str
    features: dict
    scene: str
    params: dict
    success_rate: float
    trials: int
    rounds: int
    recording: str = ""  # folder with steps.parquet, meta.json, video.mp4 of a successful run
    created: str = ""


FAMILY = {"box": "flat_sided", "cylinder": "round", "sphere": "round", "capsule": "elongated"}


def distance(a: dict, b: dict, scene_a: str, scene_b: str) -> float:
    """0 = same kind of object in the same scene. Different shape costs 0.5 within a family (e.g. round), else 1."""
    d = 0.0 if a["shape"] == b["shape"] else (0.5 if FAMILY[a["shape"]] == FAMILY[b["shape"]] else 1.0)
    d += 0.0 if scene_a == scene_b else 1.0
    d += abs(a["width_m"] - b["width_m"]) / 0.03
    d += abs(a["height_m"] - b["height_m"]) / 0.06
    return d


class SuccessLibrary:
    def __init__(self, path: Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.memories: list[Memory] = []
        if self.path.exists():
            self.memories = [Memory(**m) for m in json.loads(self.path.read_text())]

    def lookup(self, features: dict, scene: str) -> tuple[Memory, float] | None:
        if not self.memories:
            return None
        best = min(self.memories, key=lambda m: distance(features, m.features, scene, m.scene))
        d = distance(features, best.features, scene, best.scene)
        return (best, d) if d <= MATCH_THRESHOLD else None

    def save(self, memory: Memory) -> None:
        memory.created = memory.created or time.strftime("%Y-%m-%d %H:%M:%S")
        self.memories = [m for m in self.memories if m.task != memory.task] + [memory]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(m) for m in self.memories], indent=2))

    def __len__(self) -> int:
        return len(self.memories)
