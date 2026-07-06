"""Paired significance utilities."""

from __future__ import annotations

from math import erfc, sqrt


def fix_break_counts(base_correct: list[bool], new_correct: list[bool]) -> dict[str, int]:
    if len(base_correct) != len(new_correct):
        raise ValueError("paired lists must have the same length")
    fix = sum((not b) and n for b, n in zip(base_correct, new_correct))
    break_ = sum(b and (not n) for b, n in zip(base_correct, new_correct))
    same = len(base_correct) - fix - break_
    return {"fix": fix, "break": break_, "same": same}


def mcnemar_midp(base_correct: list[bool], new_correct: list[bool]) -> float:
    counts = fix_break_counts(base_correct, new_correct)
    b, c = counts["break"], counts["fix"]
    n = b + c
    if n == 0:
        return 1.0
    z = (abs(b - c) - 1.0) / sqrt(n)
    return erfc(abs(z) / sqrt(2.0))

