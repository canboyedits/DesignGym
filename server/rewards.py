from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence


EPS = 1e-9


REGION_CENTERS = {
    "top_band": (0.50, 0.10),
    "hero_center": (0.50, 0.48),
    "safe_lower_right": (0.78, 0.82),
    "top_right": (0.84, 0.12),
    "right_column": (0.80, 0.50),
    "left_column": (0.22, 0.48),
    "lower_left": (0.28, 0.76),
    "lower_right": (0.72, 0.76),
    "middle_band": (0.50, 0.55),
    "upper_right": (0.78, 0.28),
    "footer_strip": (0.50, 0.90),
    "footer_left": (0.18, 0.90),
}


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def center(bbox: Sequence[float]) -> tuple[float, float]:
    return float(bbox[0]) + float(bbox[2]) / 2.0, float(bbox[1]) + float(bbox[3]) / 2.0


def instruction_score(elements: List[Dict[str, Any]], brief: Mapping[str, Any]) -> float:
    required = brief.get("required_regions", {}) or {}
    if not required:
        return 0.0

    by_id = {str(e["id"]): e for e in elements}
    scores: List[float] = []

    for element_id, region_id in required.items():
        element = by_id.get(str(element_id))
        target = REGION_CENTERS.get(str(region_id))

        if not element or not target:
            scores.append(0.0)
            continue

        cx, cy = center(element["bbox"])
        dist = math.sqrt((cx - target[0]) ** 2 + (cy - target[1]) ** 2)

        # Smooth score: near target ≈ 1, far target ≈ 0.
        scores.append(math.exp(-dist / 0.22))

    return clamp(sum(scores) / max(1, len(scores)))


def instruction_gap_report(elements: List[Dict[str, Any]], brief: Mapping[str, Any]) -> List[str]:
    required = brief.get("required_regions", {}) or {}
    by_id = {str(e["id"]): e for e in elements}
    out: List[str] = []

    for element_id, region_id in required.items():
        element = by_id.get(str(element_id))
        target = REGION_CENTERS.get(str(region_id))
        if not element or not target:
            out.append(f"{element_id} is missing or has no target region.")
            continue

        cx, cy = center(element["bbox"])
        dist = math.sqrt((cx - target[0]) ** 2 + (cy - target[1]) ** 2)
        if dist > 0.25:
            out.append(f"{element_id} is far from required region {region_id}.")

    return out


def critic_feedback(
    metrics: Mapping[str, float],
    elements: List[Dict[str, Any]],
    brief: Mapping[str, Any],
    instruction: float,
    phase: str,
) -> List[str]:
    feedback: List[str] = []

    feedback.extend(instruction_gap_report(elements, brief))

    if instruction < 0.55:
        feedback.append("Brief satisfaction is weak; prioritize required element-region placement.")

    if metrics.get("hierarchy", 1.0) < 0.60:
        feedback.append("Visual hierarchy is weak; promote the most important text.")

    if metrics.get("spacing", 1.0) < 0.60:
        feedback.append("Spacing rhythm is weak; align or distribute related elements.")

    if metrics.get("occupancy", 1.0) < 0.50:
        feedback.append("Layout occupancy is off target; resize key content blocks.")

    if phase == "structure":
        feedback.append("Begin with global structure before local refinement.")

    if phase == "placement":
        feedback.append("Place key elements into their target semantic regions.")

    if phase == "polish":
        feedback.append("Use small corrective edits; avoid large structural changes.")

    seen = set()
    clean: List[str] = []
    for item in feedback:
        if item not in seen:
            clean.append(item)
            seen.add(item)

    return clean[:5]


def compose_reward(
    *,
    layout_delta: float,
    best_score_delta: float,
    instruction_progress: float,
    phase_correctness: float,
    validity_score: float,
    final_success_bonus: float,
    no_op_penalty: float,
    oscillation_penalty: float,
    early_finalize_penalty: float,
) -> Dict[str, float]:
    positive = (
        0.25 * max(0.0, layout_delta)
        + 0.15 * max(0.0, best_score_delta)
        + 0.20 * max(0.0, instruction_progress)
        + 0.15 * clamp(phase_correctness)
        + 0.10 * clamp(validity_score)
        + 0.10 * clamp(final_success_bonus)
    )

    penalties = (
        max(0.0, no_op_penalty)
        + max(0.0, oscillation_penalty)
        + max(0.0, early_finalize_penalty)
    )

    total = clamp(positive - penalties)

    return {
        "layout_delta": round(float(layout_delta), 6),
        "best_score_delta": round(float(best_score_delta), 6),
        "instruction_progress": round(float(instruction_progress), 6),
        "phase_correctness": round(float(phase_correctness), 6),
        "validity_score": round(float(validity_score), 6),
        "final_success_bonus": round(float(final_success_bonus), 6),
        "no_op_penalty": round(float(no_op_penalty), 6),
        "oscillation_penalty": round(float(oscillation_penalty), 6),
        "early_finalize_penalty": round(float(early_finalize_penalty), 6),
        "total": round(float(total), 6),
    }