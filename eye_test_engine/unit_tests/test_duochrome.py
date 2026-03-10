"""Unit tests for modules.duochrome.DuochromeModule."""
import pytest
from conftest import make_row
from eye_test_engine.modules.duochrome import DuochromeModule


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def module(thresholds):
    return DuochromeModule(thresholds)


# ── analyze_response ─────────────────────────────────────────────────────

class TestAnalyzeResponse:
    """Tests for DuochromeModule.analyze_response."""

    def test_red_response(self, module):
        """'Red clearer' -> patient is slightly myopic, add +0.25D."""
        row = make_row(
            occluder="Left_Occluded",
            chart_display="duochrome",
            intent="Red is clearer",
        )
        result = module.analyze_response(row)

        assert result["response"] == "red"
        assert "+0.25D" in result["recommendation"]
        assert result["eye"] == "right"

    def test_green_response(self, module):
        """'Green clearer' -> patient is slightly hyperopic, add -0.25D."""
        row = make_row(
            occluder="Right_Occluded",
            chart_display="duochrome",
            intent="Green is clearer",
        )
        result = module.analyze_response(row)

        assert result["response"] == "green"
        assert "-0.25D" in result["recommendation"]
        assert result["eye"] == "left"

    def test_both_same_response(self, module):
        """'Both same' -> balanced, no adjustment."""
        row = make_row(
            occluder="Left_Occluded",
            chart_display="duochrome",
            intent="Both same",
        )
        result = module.analyze_response(row)

        assert result["response"] == "both_same"
        assert "Balanced" in result["recommendation"]

    def test_unknown_intent(self, module):
        """Unrecognized intent -> 'unknown'."""
        row = make_row(
            occluder="Left_Occluded",
            chart_display="duochrome",
            intent="I don't know",
        )
        result = module.analyze_response(row)

        assert result["response"] == "unknown"
        assert "Unable to determine" in result["recommendation"]

    def test_none_intent_no_crash(self, module):
        """None intent should not raise an exception."""
        row = make_row(
            occluder="Left_Occluded",
            chart_display="duochrome",
            intent=None,
        )
        result = module.analyze_response(row)
        assert result["response"] == "unknown"

    # ── Occluder → eye mapping ───────────────────────────────────────────

    def test_left_occluded_means_right_eye(self, module):
        row = make_row(occluder="Left_Occluded", intent="Red is clearer")
        assert module.analyze_response(row)["eye"] == "right"

    def test_right_occluded_means_left_eye(self, module):
        row = make_row(occluder="Right_Occluded", intent="Red is clearer")
        assert module.analyze_response(row)["eye"] == "left"

    def test_bino_means_both(self, module):
        row = make_row(occluder="BINO", intent="Red is clearer")
        assert module.analyze_response(row)["eye"] == "both"

    @pytest.mark.parametrize("intent,expected", [
        ("Red is clearer", "red"),
        ("red", "red"),
        ("Green is clearer", "green"),
        ("green", "green"),
        ("Both are the same", "both_same"),
        ("same", "both_same"),
    ])
    def test_various_intent_strings(self, module, intent, expected):
        """Multiple phrasing variants should map correctly."""
        row = make_row(occluder="Left_Occluded", intent=intent)
        assert module.analyze_response(row)["response"] == expected


# ── is_complete ──────────────────────────────────────────────────────────

class TestIsComplete:
    """Tests for DuochromeModule.is_complete."""

    def test_with_intent(self, module):
        row = make_row(intent="Red is clearer")
        assert module.is_complete(row) is True

    def test_none_intent(self, module):
        row = make_row(intent=None)
        assert module.is_complete(row) is False

    def test_empty_string_intent(self, module):
        row = make_row(intent="")
        assert module.is_complete(row) is False

    def test_whitespace_only_intent(self, module):
        row = make_row(intent="   ")
        assert module.is_complete(row) is False
