from __future__ import annotations

import copy
import random
from typing import Any, Dict


BRIEF_LIBRARY: Dict[str, list[Dict[str, Any]]] = {
    "poster_basic_v1": [
        {
            "brief_id": "luxury_event_poster",
            "domain": "graphic_layout",
            "style": "luxury_minimal",
            "primary_goal": "Make the title and hero image dominant while keeping the CTA visible.",
            "required_regions": {
                "title": "top_band",
                "subtitle": "top_band",
                "hero_image": "hero_center",
                "cta": "safe_lower_right",
                "logo": "top_right",
            },
            "preferences": {
                "whitespace": "high",
                "hierarchy": "strong",
                "cta_visibility": "high",
            },
        },
        {
            "brief_id": "product_launch_poster",
            "domain": "graphic_layout",
            "style": "clean_commercial",
            "primary_goal": "Create a clean product poster with clear brand, hero, and CTA hierarchy.",
            "required_regions": {
                "title": "top_band",
                "hero_image": "hero_center",
                "cta": "safe_lower_right",
                "logo": "top_right",
            },
            "preferences": {
                "whitespace": "medium",
                "hierarchy": "strong",
                "cta_visibility": "high",
            },
        },
    ],
    "editorial_cover_v1": [
        {
            "brief_id": "editorial_feature_cover",
            "domain": "editorial_layout",
            "style": "magazine_editorial",
            "primary_goal": "Preserve masthead dominance and create a readable headline stack.",
            "required_regions": {
                "masthead": "top_band",
                "hero_image": "hero_center",
                "headline_1": "lower_left",
                "headline_2": "lower_left",
                "headline_3": "lower_left",
                "barcode": "footer_strip",
            },
            "preferences": {
                "reading_order": "strict",
                "hierarchy": "strong",
                "balance": "editorial",
            },
        }
    ],
    "dense_flyer_v1": [
        {
            "brief_id": "local_offer_flyer",
            "domain": "dense_layout",
            "style": "dense_commercial",
            "primary_goal": "Keep many elements readable while making price and CTA easy to find.",
            "required_regions": {
                "title": "top_band",
                "details": "middle_band",
                "price_badge": "upper_right",
                "cta": "safe_lower_right",
                "sponsor_strip": "footer_strip",
            },
            "preferences": {
                "density": "high",
                "spacing": "controlled",
                "cta_visibility": "high",
            },
        }
    ],
}


def choose_brief(task_id: str, seed: int = 0) -> Dict[str, Any]:
    briefs = BRIEF_LIBRARY.get(task_id) or BRIEF_LIBRARY["poster_basic_v1"]
    rng = random.Random(f"{task_id}:{seed}")
    return copy.deepcopy(rng.choice(briefs))