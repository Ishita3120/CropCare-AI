"""core/knowledge_base.py -- loads the crop/disease -> treatment lookup table."""
from __future__ import annotations

import json
import os
from typing import Optional

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "treatments.json")

with open(_DATA_PATH, "r") as f:
    _DATA = json.load(f)


def lookup_treatment(crop: str, disease: str) -> Optional[dict]:
    return _DATA.get(crop, {}).get(disease)
