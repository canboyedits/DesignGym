from __future__ import annotations

from typing import List


PHASES = ["structure", "placement", "refinement", "polish", "finalize"]

ALLOWED_ACTIONS_BY_PHASE = {
    "structure": ["apply_template"],
    "placement": ["anchor_to_region", "reflow_group", "align"],
    "refinement": ["move", "resize", "align", "distribute", "promote", "reflow_group"],
    "polish": ["move", "align", "distribute", "promote", "snap"],
    "finalize": ["finalize"],
}


def get_phase(step_count: int, max_steps: int, current_score: float, done: bool = False) -> str:
    if done:
        return "finalize"

    progress = step_count / max(1, max_steps)

    if progress < 0.18:
        return "structure"
    if progress < 0.42:
        return "placement"
    if progress < 0.75:
        return "refinement"
    if progress < 0.95:
        return "polish"

    return "finalize"


def allowed_actions_for_phase(phase: str) -> List[str]:
    return list(ALLOWED_ACTIONS_BY_PHASE.get(phase, []))


def phase_score_for_action(action_type: str, phase: str) -> float:
    allowed = ALLOWED_ACTIONS_BY_PHASE.get(phase, [])

    if action_type in allowed:
        return 1.0

    # Soft allowances keep learning dense instead of turning everything into zero reward.
    if phase == "structure" and action_type in {"anchor_to_region", "reflow_group", "align"}:
        return 0.45

    if phase == "placement" and action_type in {"move", "resize", "promote"}:
        return 0.50

    if phase == "refinement" and action_type in {"anchor_to_region", "snap"}:
        return 0.55

    if phase == "polish" and action_type in {"resize", "anchor_to_region"}:
        return 0.45

    if phase != "finalize" and action_type == "finalize":
        return 0.0

    return 0.15