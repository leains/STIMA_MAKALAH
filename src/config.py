"""Configuration for one analysis run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Analysis settings
@dataclass(frozen=True)
class AnalysisConfig:
    """Store the place, travel limit, business category, and output folder."""

    place: str = "Coblong, Bandung, Indonesia"
    network_type: str = "walk"
    cutoff_m: float = 5000.0
    competitor_category: str = "cafe"
    output_dir: str = "outputs"
    candidate_lat: float | None = None
    candidate_lon: float | None = None
    edge_buffer_m: float = 35.0


# Business tags
# Keep this list small so Overpass requests stay reasonable.
BUSINESS_TAGS: dict[str, Any] = {
    "amenity": ["cafe", "restaurant", "fast_food", "pharmacy", "bank", "atm"],
    "shop": ["convenience", "supermarket", "bakery", "laundry", "clothes", "mall"],
    "office": True,
}

# Customer proxy tags
RESIDENTIAL_BUILDING_TAGS: dict[str, Any] = {
    "building": ["house", "residential", "apartments", "detached", "terrace", "dormitory"]
}

# Direct competitor categories
COMPETITOR_TAG_LOOKUP: dict[str, dict[str, Any]] = {
    "cafe": {"amenity": "cafe"},
    "restaurant": {"amenity": "restaurant"},
    "fast_food": {"amenity": "fast_food"},
    "pharmacy": {"amenity": "pharmacy"},
    "bank": {"amenity": "bank"},
    "convenience": {"shop": "convenience"},
    "supermarket": {"shop": "supermarket"},
    "bakery": {"shop": "bakery"},
    "laundry": {"shop": "laundry"},
}
