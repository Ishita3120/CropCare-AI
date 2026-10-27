"""
core/decision_engine.py

Rule-based, auditable treat/no-treat + dosage decision.

Combines:
- severity
- selected crop/disease treatment knowledge
- optional environmental conditions

The decision is transparent and every result includes a plain-language reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.severity import SeverityResult


# ============================================================
# CONSTANTS
# ============================================================

NO_TREAT_BELOW_SEVERITY = 0.03

HUMIDITY_RISK_PCT = 80
SOIL_MOISTURE_RISK_PCT = 60

BASE_SPRAY_DURATION_S = 3.0
MAX_SPRAY_DURATION_S = 15.0

ML_PER_SECOND = 2.0


# ============================================================
# DECISION RESULT
# ============================================================

@dataclass
class DecisionResult:
    treat: bool
    reason: str
    spray_duration_s: float
    spray_volume_ml: float
    recheck_days: int
    risk_note: str


# ============================================================
# DECISION ENGINE
# ============================================================

def decide(
    severity: SeverityResult,
    disease_label: str,
    treatment: Optional[dict],
    temperature_c: Optional[float] = None,
    humidity_pct: Optional[float] = None,
    soil_moisture_pct: Optional[float] = None,
) -> DecisionResult:

    # ========================================================
    # 1. HEALTHY LEAF
    # ========================================================

    if disease_label == "healthy":

        return DecisionResult(
            treat=False,
            reason="No disease signal detected on this leaf.",
            spray_duration_s=0.0,
            spray_volume_ml=0.0,
            recheck_days=7,
            risk_note="No environmental risk scaling applied.",
        )

    # ========================================================
    # 2. VERY LOW AFFECTED AREA
    # ========================================================

    if severity.score < NO_TREAT_BELOW_SEVERITY:

        recheck_days = (
            treatment.get("recheck_days", 7)
            if treatment
            else 7
        )

        return DecisionResult(
            treat=False,
            reason=(
                f"Affected area ({severity.score:.1%}) is below the "
                f"treatment threshold ({NO_TREAT_BELOW_SEVERITY:.0%}). "
                f"Spraying now would be unnecessary chemical use."
            ),
            spray_duration_s=0.0,
            spray_volume_ml=0.0,
            recheck_days=recheck_days,
            risk_note="No environmental risk scaling applied.",
        )

    # ========================================================
    # 3. NO TREATMENT INFORMATION
    # ========================================================

    if treatment is None:

        return DecisionResult(
            treat=False,
            reason=(
                f"Affected area ({severity.score:.1%}) is above the "
                f"treatment threshold ({NO_TREAT_BELOW_SEVERITY:.0%}), "
                f"but no treatment information was found for "
                f"{disease_label}."
            ),
            spray_duration_s=0.0,
            spray_volume_ml=0.0,
            recheck_days=7,
            risk_note="Treatment information is unavailable.",
        )

    # ========================================================
    # 4. CHECK WHETHER THE KNOWLEDGE BASE ALLOWS SPRAYING
    # ========================================================
    #
    # This is important.
    #
    # If the KB contains:
    #
    # min_dosage_ml_per_l = 0
    # max_dosage_ml_per_l = 0
    #
    # then this entry must NOT result in an automatic spray.
    #
    # It can still provide guidance/advice, but the actuator
    # must remain OFF.
    # ========================================================

    min_dose = float(
        treatment.get("min_dosage_ml_per_l", 0) or 0
    )

    max_dose = float(
        treatment.get("max_dosage_ml_per_l", 0) or 0
    )

    if min_dose <= 0 and max_dose <= 0:

        treatment_name = treatment.get(
            "name",
            "Treatment guidance"
        )

        advice = treatment.get(
            "advice",
            "Follow the applicable registered treatment label."
        )

        return DecisionResult(
            treat=False,
            reason=(
                f"Severity is {severity.severity_class} "
                f"({severity.score:.1%} of the leaf affected). "
                f"{treatment_name} is listed in the knowledge base, "
                f"but no spray dosage is configured for this treatment. "
                f"Actuation is therefore disabled. {advice}"
            ),
            spray_duration_s=0.0,
            spray_volume_ml=0.0,
            recheck_days=treatment.get(
                "recheck_days",
                3
            ),
            risk_note=(
                "No spray applied because the knowledge base "
                "contains a zero dosage range."
            ),
        )

    # ========================================================
    # 5. ENVIRONMENTAL RISK
    # ========================================================

    # Missing environmental readings use neutral values.

    humidity_pct = (
        50.0
        if humidity_pct is None
        else humidity_pct
    )

    soil_moisture_pct = (
        40.0
        if soil_moisture_pct is None
        else soil_moisture_pct
    )

    multiplier = 1.0

    risk_bits = []

    # --------------------------------------------------------
    # High humidity
    # --------------------------------------------------------

    if humidity_pct >= HUMIDITY_RISK_PCT:

        multiplier += 0.25

        risk_bits.append(
            f"high humidity ({humidity_pct:.0f}%)"
        )

    # --------------------------------------------------------
    # High soil moisture
    # --------------------------------------------------------

    if soil_moisture_pct >= SOIL_MOISTURE_RISK_PCT:

        multiplier += 0.15

        risk_bits.append(
            f"high soil moisture ({soil_moisture_pct:.0f}%)"
        )

    # ========================================================
    # 6. RISK NOTE
    # ========================================================

    if risk_bits:

        risk_note = (
            f"Dose scaled up for {', '.join(risk_bits)} "
            f"-- these conditions favor disease spread."
        )

    else:

        risk_note = (
            "Environmental conditions are nominal; "
            "no dose scaling applied."
        )

    # ========================================================
    # 7. SPRAY DURATION
    # ========================================================
    #
    # The current prototype uses severity + environmental
    # conditions to determine actuator duration.
    #
    # The KB dosage range is still displayed separately as
    # the treatment dosage range.
    # ========================================================

    duration = round(
        min(
            BASE_SPRAY_DURATION_S
            * (0.5 + severity.score)
            * multiplier,
            MAX_SPRAY_DURATION_S,
        ),
        2,
    )

    # Never allow a negative/zero actuator duration for a
    # treatment that actually has a positive dosage range.

    duration = max(
        duration,
        0.1
    )

    volume = round(
        duration * ML_PER_SECOND,
        1,
    )

    # ========================================================
    # 8. TREATMENT RECOMMENDATION
    # ========================================================

    treatment_name = treatment.get(
        "name",
        "Treatment"
    )

    reason = (
        f"Severity is {severity.severity_class} "
        f"({severity.score:.1%} of the leaf affected). "
        f"Recommending {treatment_name} within the configured "
        f"treatment guidance."
    )

    # ========================================================
    # 9. FINAL DECISION
    # ========================================================

    return DecisionResult(
        treat=True,
        reason=reason,
        spray_duration_s=duration,
        spray_volume_ml=volume,
        recheck_days=treatment.get(
            "recheck_days",
            3
        ),
        risk_note=risk_note,
    )