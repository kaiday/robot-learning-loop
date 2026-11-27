"""Run again, and check it really fails less: a regression gate over held-out seeds."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from rll.analyze import success_rate
from rll.runner import EpisodeResult, run_batch


@dataclass
class GateResult:
    accepted: bool
    base_rate: float
    cand_rate: float
    n: int
    fixed: int  # seeds that failed before and pass now
    regressions: int  # seeds that passed before and fail now
    retry_fixed: int  # of this iteration's training failures, how many the candidate now passes
    retry_total: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def run_gate(
    base_eval: list[EpisodeResult],
    cand_params: dict,
    eval_seeds: list[int],
    failed_train_seeds: list[int],
    min_gain: float = 0.01,
    max_regress_frac: float = 0.03,
) -> tuple[GateResult, list[EpisodeResult]]:
    cand_eval = run_batch(eval_seeds, cand_params)
    base = {r.cfg.seed: r.success for r in base_eval}
    cand = {r.cfg.seed: r.success for r in cand_eval}
    fixed = sum(1 for s in eval_seeds if cand[s] and not base[s])
    regress = sum(1 for s in eval_seeds if base[s] and not cand[s])
    retry = run_batch(failed_train_seeds, cand_params) if failed_train_seeds else []
    b, c = success_rate(base_eval), success_rate(cand_eval)

    if c < b + min_gain:
        accepted, reason = False, f"no gain: {b:.0%} -> {c:.0%}"
    elif regress > max(1, int(max_regress_frac * len(eval_seeds))):
        accepted, reason = False, f"{regress} regressions on previously passing seeds"
    else:
        accepted, reason = True, f"{b:.0%} -> {c:.0%}, fixed {fixed}, regressed {regress}"
    result = GateResult(
        accepted=accepted, base_rate=b, cand_rate=c, n=len(eval_seeds), fixed=fixed, regressions=regress,
        retry_fixed=int(sum(r.success for r in retry)), retry_total=len(retry), reason=reason,
    )
    return result, cand_eval
