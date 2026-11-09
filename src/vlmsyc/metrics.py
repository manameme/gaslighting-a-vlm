"""Metrics. For now only the flip-rate summary used by the pilot (Phase 3).

EIR, ECR, AUROC and the gates come in Phase 5.
"""

from __future__ import annotations


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def flip_summary(rows: list[dict]) -> dict:
    """Turn-1 accuracy, flip rates (overall and split by a1 correct/wrong) and yes/no mass.

    Turn-1 accuracy is computed per question (not per row), because every
    pushback row of a question repeats the same turn-1 answer.
    """
    first_by_uid = {r["uid"]: r for r in rows}
    correct = [r for r in rows if r["a1_correct"]]
    wrong = [r for r in rows if not r["a1_correct"]]
    return {
        "n_questions": len(first_by_uid),
        "n_rows": len(rows),
        "turn1_acc": _mean([float(r["a1_correct"]) for r in first_by_uid.values()]),
        "flip_rate": _mean([float(r["flip"]) for r in rows]),
        "flip_rate_a1_correct": _mean([float(r["flip"]) for r in correct]),
        "flip_rate_a1_wrong": _mean([float(r["flip"]) for r in wrong]),
        "turn2_acc": _mean([float(r["a2_correct"]) for r in rows]),
        "mean_mass_t1": _mean([r["yesno_mass_t1"] for r in first_by_uid.values()]),
        "mean_mass_t2": _mean([r["yesno_mass_t2"] for r in rows]),
        "min_mass_t2": min((r["yesno_mass_t2"] for r in rows), default=None),
    }
