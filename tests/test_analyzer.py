"""Smoke tests for the analysis pipeline. Run with: pytest tests/"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from core.image_analyzer import analyze_image
from core.severity import estimate_severity
from core.knowledge_base import lookup_treatment
from core.decision_engine import decide


def _solid_image(color, size=(200, 200)):
    return Image.new("RGB", size, color)


def test_healthy_green_leaf_detected_as_healthy():
    img = _solid_image((60, 140, 70))  # solid green
    result = analyze_image(img, "tomato")
    assert result.disease_label == "healthy"
    assert result.affected_area_ratio < 0.1


def test_brown_leaf_detected_as_diseased():
    img = _solid_image((110, 70, 40))  # brown
    result = analyze_image(img, "tomato")
    assert result.disease_label in ("early_blight", "late_blight")
    assert result.affected_area_ratio > 0.1


def test_wheat_only_offers_leaf_rust():
    img = _solid_image((120, 90, 30))
    result = analyze_image(img, "wheat")
    assert result.disease_label in ("healthy", "leaf_rust")


def test_severity_thresholds():
    assert estimate_severity(0.02).severity_class == "low"
    assert estimate_severity(0.20).severity_class == "medium"
    assert estimate_severity(0.60).severity_class == "high"


def test_knowledge_base_lookup():
    entry = lookup_treatment("tomato", "early_blight")
    assert entry is not None
    assert entry["active_ingredient"] == "Chlorothalonil"


def test_decision_no_treat_for_healthy():
    sev = estimate_severity(0.01)
    treatment = lookup_treatment("tomato", "healthy")
    result = decide(sev, "healthy", treatment)
    assert result.treat is False


def test_decision_treats_high_severity():
    sev = estimate_severity(0.45)
    treatment = lookup_treatment("tomato", "early_blight")
    result = decide(sev, "early_blight", treatment, humidity_pct=90, soil_moisture_pct=70)
    assert result.treat is True
    assert result.spray_duration_s > 0
    assert "humidity" in result.risk_note


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
