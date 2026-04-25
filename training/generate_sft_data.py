from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from models import DesignGymAction
    from server.DesignGym_environment import DesignGymEnvironment
except Exception:
    from DesignGym.models import DesignGymAction
    from DesignGym.server.DesignGym_environment import DesignGymEnvironment


ID_RE = re.compile(r"([A-Za-z0-9_]+)@\(")

TASKS = [
    "poster_basic_v1",
    "editorial_cover_v1",
    "dense_flyer_v1",
]


SYSTEM_PROMPT = (
    "You are a long-horizon spatial layout design agent. "
    "You receive a design brief, current phase, layout state, metrics, and feedback. "
    "Output exactly one valid minified JSON action object and nothing else."
)


def compact_action(action: DesignGymAction) -> str:
    """Return concise JSON action for SFT target."""
    data = action.model_dump(exclude_none=True)

    # Drop empty/default fields to avoid teaching noisy long JSON.
    cleaned: Dict[str, Any] = {}
    for key, value in data.items():
        if key == "element_ids" and value == []:
            continue
        if key in {"dx", "dy", "dw", "dh", "strength"} and float(value) == 0.0:
            continue
        if key == "grid" and int(value) == 0:
            continue
        if key == "anchor" and value == "center" and data.get("action_type") != "resize":
            continue
        cleaned[key] = value

    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def ids_in_layout(obs: Any) -> List[str]:
    return ID_RE.findall(getattr(obs, "layout_summary", "") or "")


def has_id(obs: Any, element_id: str) -> bool:
    return element_id in ids_in_layout(obs)


def task_kind(task_id: str) -> str:
    if "editorial" in task_id:
        return "editorial"
    if "dense" in task_id:
        return "dense"
    return "poster"


def make_template_actions(task_id: str) -> List[DesignGymAction]:
    kind = task_kind(task_id)
    if kind == "poster":
        return [
            DesignGymAction(action_type="apply_template", template_id="hero"),
            DesignGymAction(action_type="apply_template", template_id="split"),
        ]
    if kind == "editorial":
        return [
            DesignGymAction(action_type="apply_template", template_id="editorial"),
            DesignGymAction(action_type="apply_template", template_id="grid"),
        ]
    return [
        DesignGymAction(action_type="apply_template", template_id="grid"),
        DesignGymAction(action_type="apply_template", template_id="hero"),
    ]


def candidate_actions(obs: Any, recent_actions: Sequence[str]) -> List[DesignGymAction]:
    task_id = getattr(obs, "task_id", "")
    kind = task_kind(task_id)
    phase = getattr(obs, "phase", "refinement")
    worst = set(getattr(obs, "worst_metrics", []) or [])
    brief = getattr(obs, "brief", {}) or {}
    required_regions = brief.get("required_regions", {}) or {}
    instruction = float(getattr(obs, "instruction_score", 0.0) or 0.0)

    actions: List[DesignGymAction] = []

    # Structure phase: make global layout choice early.
    if int(getattr(obs, "step_count", 0) or 0) == 0 or phase == "structure":
        actions.extend(make_template_actions(task_id))

    # Placement: satisfy brief-required regions.
    priority = [
        "cta",
        "price_badge",
        "hero_image",
        "masthead",
        "title",
        "subtitle",
        "headline_1",
        "headline_2",
        "headline_3",
        "details",
        "sponsor_strip",
        "logo",
    ]

    if instruction < 0.85 or phase in {"placement", "structure"}:
        for element_id in priority:
            if element_id in required_regions and has_id(obs, element_id):
                actions.append(
                    DesignGymAction(
                        action_type="anchor_to_region",
                        element_id=element_id,
                        region_id=str(required_regions[element_id]),
                        mode="center",
                    )
                )

    # Occupancy / text fit repair.
    if "occupancy" in worst or "text_fit" in worst:
        for element_id, dw, dh in [
            ("hero_image", 0.03, 0.02),
            ("details", 0.02, 0.02),
            ("image_left", 0.02, 0.02),
            ("image_right", 0.02, 0.02),
            ("subtitle", 0.02, 0.01),
            ("headline_2", 0.02, 0.01),
        ]:
            if has_id(obs, element_id):
                actions.append(
                    DesignGymAction(
                        action_type="resize",
                        element_id=element_id,
                        dw=dw,
                        dh=dh,
                        anchor="center",
                    )
                )

    # Hierarchy repair.
    if "hierarchy" in worst or phase in {"refinement", "polish"}:
        for element_id in [
            "title",
            "headline_1",
            "masthead",
            "cta",
            "price_badge",
            "details",
        ]:
            if has_id(obs, element_id):
                actions.append(
                    DesignGymAction(
                        action_type="promote",
                        element_id=element_id,
                        strength=0.04,
                    )
                )

    # Alignment repair.
    if "alignment" in worst or phase in {"placement", "polish"}:
        # Safe alignment candidates are useful even when alignment is not the worst metric.
        # This improves SFT coverage for long-horizon refinement behavior.
        if kind == "poster":
            ids = [x for x in ["title", "subtitle"] if has_id(obs, x)]
            if len(ids) >= 2:
                actions.append(
                    DesignGymAction(
                        action_type="align",
                        element_ids=ids,
                        axis="x",
                        mode="left",
                    )
                )

        elif kind == "editorial":
            ids = [x for x in ["masthead", "headline_1", "headline_2"] if has_id(obs, x)]
            if len(ids) >= 2:
                actions.append(
                    DesignGymAction(
                        action_type="align",
                        element_ids=ids,
                        axis="x",
                        mode="left",
                    )
                )

        else:
            ids = [x for x in ["caption_1", "caption_2"] if has_id(obs, x)]
            if len(ids) >= 2:
                actions.append(
                    DesignGymAction(
                        action_type="align",
                        element_ids=ids,
                        axis="y",
                        mode="top",
                    )
                )
    # Spacing / reading order repair.
    if "spacing" in worst or "reading_order" in worst or phase == "refinement":
        if kind == "poster":
            actions.append(DesignGymAction(action_type="reflow_group", group_id="headline", pattern="stack"))
        elif kind == "editorial":
            actions.append(DesignGymAction(action_type="reflow_group", group_id="stories", pattern="stack"))
        else:
            actions.append(DesignGymAction(action_type="reflow_group", group_id="support", pattern="row"))

    # Small polish moves.
       # Small local movement candidates.
    # These are important for teaching fine-grained spatial correction.
    if phase in {"refinement", "polish"} or "balance" in worst or "spacing" in worst:
        for element_id in [
            "hero_image",
            "title",
            "subtitle",
            "cta",
            "masthead",
            "headline_1",
            "headline_2",
            "details",
            "price_badge",
        ]:
            if has_id(obs, element_id):
                actions.append(
                    DesignGymAction(
                        action_type="move",
                        element_id=element_id,
                        dx=0.01,
                        dy=-0.01,
                    )
                )
                actions.append(
                    DesignGymAction(
                        action_type="move",
                        element_id=element_id,
                        dx=-0.01,
                        dy=0.01,
                    )
                )
                break

    # Finalize only when plausibly ready.
    score = float(getattr(obs, "current_score", 0.0) or 0.0)
    step_count = int(getattr(obs, "step_count", 0) or 0)
    max_steps = int(getattr(obs, "max_steps", 1) or 1)
    if step_count >= int(0.75 * max_steps) and score >= 0.72 and instruction >= 0.60:
        actions.append(DesignGymAction(action_type="finalize"))

    # Deduplicate.
    dedup: List[DesignGymAction] = []
    seen = set()
    for action in actions:
        key = compact_action(action)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(action)

    # Avoid repeating identical action if possible.
    filtered = [a for a in dedup if compact_action(a) not in set(recent_actions[-2:])]
    return filtered or dedup or [DesignGymAction(action_type="finalize")]


def evaluate_candidate(env: DesignGymEnvironment, action: DesignGymAction) -> float:
    """Score a candidate by simulating it on a copied environment."""
    try:
        tmp = copy.deepcopy(env)
        obs = tmp.step(action)
        state = tmp.state

        if getattr(obs, "last_action_error", None):
            return -10.0

        reward = float(getattr(state, "last_reward", 0.0) or 0.0)
        layout_score = float(getattr(state, "current_score", 0.0) or 0.0)
        instruction = float(getattr(state, "instruction_score", 0.0) or 0.0)
        phase_score = float(getattr(state, "phase_score", 0.0) or 0.0)

        # Reward is primary. Small tie-breaks prefer better final state.
        return reward + 0.05 * instruction + 0.03 * layout_score + 0.02 * phase_score
    except Exception:
        return -10.0


def preferred_action_type_for_example(episode_idx: int, local_step: int, obs: Any) -> str | None:
    phase = getattr(obs, "phase", "")
    bucket = (episode_idx * 31 + local_step * 17) % 100

    # Force some safe alignment examples.
    if phase in {"placement", "refinement", "polish"} and bucket < 12:
        return "align"

    # Force some fine-grained movement examples.
    if phase in {"refinement", "polish"} and 12 <= bucket < 20:
        return "move"

    return None


def choose_expert_action(
    env: DesignGymEnvironment,
    obs: Any,
    preferred_action_type: str | None = None,
) -> DesignGymAction:
    recent = list(getattr(env.state, "action_history", []) or [])
    candidates = candidate_actions(obs, recent)

    scored = [(a, evaluate_candidate(env, a)) for a in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    ranked = [a for a, _ in scored]

    # If this example is scheduled for action diversity, choose the best candidate
    # of that type, but only if it is not terrible.
    if preferred_action_type:
        preferred = [
            (a, score)
            for a, score in scored
            if a.action_type == preferred_action_type and score > -1.0
        ]
        if preferred:
            preferred.sort(key=lambda x: x[1], reverse=True)
            return preferred[0][0]

    # Normal expert choice with small top-k diversity.
    if len(ranked) > 1:
        rng_key = (
            f"{getattr(env.state, 'seed', 0)}:"
            f"{getattr(obs, 'task_id', '')}:"
            f"{getattr(obs, 'step_count', 0)}:"
            f"{len(recent)}"
        )
        rng = random.Random(rng_key)

        if rng.random() < 0.12:
            top_k = ranked[: min(4, len(ranked))]
            return rng.choice(top_k)

    return ranked[0]


def prompt_from_obs(obs: Any) -> str:
    payload = {
        "task_id": getattr(obs, "task_id", ""),
        "step_count": getattr(obs, "step_count", 0),
        "max_steps": getattr(obs, "max_steps", 0),
        "phase": getattr(obs, "phase", ""),
        "allowed_actions": getattr(obs, "allowed_actions", []),
        "current_score": round(float(getattr(obs, "current_score", 0.0) or 0.0), 4),
        "best_score_so_far": round(float(getattr(obs, "best_score_so_far", 0.0) or 0.0), 4),
        "instruction_score": round(float(getattr(obs, "instruction_score", 0.0) or 0.0), 4),
        "phase_score": round(float(getattr(obs, "phase_score", 0.0) or 0.0), 4),
        "brief": getattr(obs, "brief", {}) or {},
        "metrics": getattr(obs, "metrics", {}) or {},
        "metric_deltas": getattr(obs, "metric_deltas", {}) or {},
        "worst_metrics": getattr(obs, "worst_metrics", []) or [],
        "focus_elements": getattr(obs, "focus_elements", []) or [],
        "critic_feedback": getattr(obs, "critic_feedback", []) or [],
        "layout_summary": getattr(obs, "layout_summary", "") or "",
    }

    action_schema = {
        "action_type": "apply_template | anchor_to_region | resize | move | align | distribute | promote | reflow_group | finalize",
        "optional_fields": {
            "template_id": "hero | split | editorial | grid | draft",
            "element_id": "single element id",
            "element_ids": "list of element ids",
            "region_id": "semantic target region",
            "group_id": "semantic group id",
            "pattern": "stack | row",
            "axis": "x | y",
            "mode": "left | center | top",
            "dx_dy_dw_dh": "small normalized floats",
            "strength": "small float for promote",
        },
    }

    return (
        "Choose the next best layout edit.\n"
        "Output only one valid minified JSON action object.\n\n"
        f"STATE:\n{json.dumps(payload, sort_keys=True)}\n\n"
        f"ACTION_SCHEMA:\n{json.dumps(action_schema, sort_keys=True)}"
    )


def make_example(obs: Any, action: DesignGymAction, metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_from_obs(obs)},
            {"role": "assistant", "content": compact_action(action)},
        ],
        **metadata,
    }


def generate_examples(
    *,
    episodes: int,
    seed: int,
    max_steps_override: int | None,
    tasks: Sequence[str],
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    examples: List[Dict[str, Any]] = []

    for episode_idx in range(episodes):
        task_id = tasks[episode_idx % len(tasks)]
        episode_seed = seed + episode_idx

        env = DesignGymEnvironment()
        obs = env.reset(task_id=task_id, seed=episode_seed)

        max_steps = max_steps_override or int(getattr(obs, "max_steps", 8) or 8)

        for local_step in range(max_steps):
            if bool(getattr(obs, "done", False)) or bool(getattr(env.state, "done", False)):
                break

            preferred_type = preferred_action_type_for_example(episode_idx, local_step, obs)
            action = choose_expert_action(env, obs, preferred_action_type=preferred_type)
            before_obs = obs

            obs = env.step(action)

            metadata = {
                "task_id": task_id,
                "episode_seed": episode_seed,
                "episode_index": episode_idx,
                "step_index": local_step,
                "expert_action": compact_action(action),
                "reward_after": round(float(getattr(env.state, "last_reward", 0.0) or 0.0), 6),
                "score_after": round(float(getattr(env.state, "current_score", 0.0) or 0.0), 6),
                "instruction_score_after": round(float(getattr(env.state, "instruction_score", 0.0) or 0.0), 6),
                "phase_after": getattr(env.state, "phase", ""),
            }

            examples.append(make_example(before_obs, action, metadata))

    rng.shuffle(examples)
    return examples


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--tasks", nargs="*", default=TASKS)
    parser.add_argument("--out", type=str, default="data/sft/designgym2_sft_train.jsonl")
    parser.add_argument("--eval-out", type=str, default="data/sft/designgym2_sft_eval.jsonl")
    parser.add_argument("--eval-ratio", type=float, default=0.10)
    args = parser.parse_args()

    examples = generate_examples(
        episodes=args.episodes,
        seed=args.seed,
        max_steps_override=args.max_steps,
        tasks=args.tasks,
    )

    split_idx = int(len(examples) * (1.0 - args.eval_ratio))
    train_rows = examples[:split_idx]
    eval_rows = examples[split_idx:]

    train_count = write_jsonl(Path(args.out), train_rows)
    eval_count = write_jsonl(Path(args.eval_out), eval_rows)

    print(f"[OK] generated total={len(examples)} train={train_count} eval={eval_count}")
    print(f"[TRAIN] {args.out}")
    print(f"[EVAL]  {args.eval_out}")

    if examples:
        print("[SAMPLE]")
        print(json.dumps(examples[0], indent=2)[:3000])


if __name__ == "__main__":
    main()