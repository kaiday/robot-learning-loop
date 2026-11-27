"""Turn a batch of labelled episodes into a failure report the brain can act on.

For each failure type we report how often it happens, where (phase), and which randomized
conditions drive it: the failure rate in the low / mid / high third of each condition.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd

from rll.runner import EpisodeResult

CONDITIONS = ["mass", "friction", "half_size", "cube_yaw_abs", "obs_noise", "cube_x", "cube_y"]


def summary_table(results: list[EpisodeResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.summary() for r in results])
    df = df.rename(columns={c: c.removeprefix("cfg_") for c in df.columns})
    df["cube_yaw_abs"] = df["cube_yaw"].abs()
    df["failure_type"] = df["failure_type"].fillna("ok")
    return df


def _drivers(df: pd.DataFrame, kind: str) -> list[dict]:
    is_fail = df.failure_type == kind
    out = []
    for c in CONDITIONS:
        try:
            bins = pd.qcut(df[c], 3, labels=["low", "mid", "high"], duplicates="drop")
        except ValueError:
            continue
        rates = is_fail.groupby(bins, observed=False).mean()
        edges = df[c].quantile([0, 1 / 3, 2 / 3, 1]).round(4).tolist()
        spread = float(rates.max() - rates.min())
        out.append(
            {
                "condition": c,
                "failure_rate_by_tercile": {k: round(float(v), 3) for k, v in rates.items()},
                "tercile_edges": edges,
                "spread": round(spread, 3),
            }
        )
    return sorted(out, key=lambda d: -d["spread"])[:3]


def build_report(results: list[EpisodeResult], params: dict) -> dict:
    df = summary_table(results)
    n = len(df)
    counts = Counter(df.failure_type)
    failures = {}
    for kind, k in counts.most_common():
        if kind == "ok":
            continue
        sub = [r for r in results if r.failure and r.failure["type"] == kind]
        failures[kind] = {
            "count": k,
            "rate": round(k / n, 3),
            "phases": dict(Counter(r.failure["phase"] for r in sub)),
            "drivers": _drivers(df, kind),
            "examples": [
                {"seed": r.cfg.seed, "event": r.failure, "conditions": _round(r.cfg.to_dict())} for r in sub[:3]
            ],
        }
    return {
        "episodes": n,
        "success_rate": round(counts["ok"] / n, 3) if n else 0.0,
        "failure_counts": {k: v for k, v in counts.items() if k != "ok"},
        "failures": failures,
        "params": params,
    }


def report_markdown(report: dict) -> str:
    lines = [
        f"# Failure report — {report['episodes']} episodes, success {report['success_rate']:.0%}",
        "",
        "## Current params",
        "```json",
        json.dumps(report["params"], indent=2),
        "```",
    ]
    for kind, f in report["failures"].items():
        lines += ["", f"## {kind}: {f['count']} episodes ({f['rate']:.0%}), phases {f['phases']}", "Top drivers:"]
        for d in f["drivers"]:
            lines.append(
                f"- `{d['condition']}` failure rate low/mid/high = "
                + " / ".join(f"{v:.0%}" for v in d["failure_rate_by_tercile"].values())
                + f" (tercile edges {d['tercile_edges']})"
            )
        lines.append("Examples:")
        for ex in f["examples"]:
            e = ex["event"]
            lines.append(f"- seed {ex['seed']}: t={e['t']:.2f}s phase={e['phase']} detail={e['detail']} conditions={ex['conditions']}")
    return "\n".join(lines) + "\n"


def _round(d: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def success_rate(results: list[EpisodeResult]) -> float:
    return float(np.mean([r.success for r in results])) if results else 0.0
