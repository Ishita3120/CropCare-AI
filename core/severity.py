"""core/severity.py -- maps affected-area ratio to a severity class."""
from __future__ import annotations

from dataclasses import dataclass

LOW_MAX = 0.10
MEDIUM_MAX = 0.30


@dataclass
class SeverityResult:
    score: float          # 0-1, same as affected_area_ratio
    severity_class: str   # 'low' | 'medium' | 'high'


def estimate_severity(affected_area_ratio: float) -> SeverityResult:
    score = round(min(max(affected_area_ratio, 0.0), 1.0), 4)
    if score <= LOW_MAX:
        severity_class = "low"
    elif score <= MEDIUM_MAX:
        severity_class = "medium"
    else:
        severity_class = "high"
    return SeverityResult(score=score, severity_class=severity_class)
