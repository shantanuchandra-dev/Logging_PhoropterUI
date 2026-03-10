"""Unit tests for modules.cylinder_axis.CylinderAxisModule."""
import pytest
from conftest import make_row
from eye_test_engine.modules.cylinder_axis import CylinderAxisModule


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def module(thresholds):
    return CylinderAxisModule(thresholds)


# ── analyze_flip_pair ────────────────────────────────────────────────────

class TestAnalyzeFlipPair:
    """Tests for CylinderAxisModule.analyze_flip_pair."""

    def test_no_next_row(self, module):
        """Missing next_row -> choice='unknown', returns current axis."""
        flip1 = make_row(r_axis=90.0)
        flip2 = make_row(r_axis=90.0)
        result = module.analyze_flip_pair(flip1, flip2, next_row=None, eye="right")

        assert result["choice"] == "unknown"
        assert result["next_axis"] == 90.0

    def test_positive_change_is_flip1_gap(self, module):
        """Axis increases -> GAP / flip1."""
        flip1 = make_row(r_axis=90.0)
        flip2 = make_row(r_axis=90.0)
        next_row = make_row(r_axis=95.0)
        result = module.analyze_flip_pair(flip1, flip2, next_row, eye="right")

        assert result["choice"] == "flip1"
        assert result["axis_change"] == pytest.approx(5.0)
        assert "GAP" in result["intent"]

    def test_negative_change_is_flip2_ram(self, module):
        """Axis decreases -> RAM / flip2."""
        flip1 = make_row(r_axis=90.0)
        flip2 = make_row(r_axis=90.0)
        next_row = make_row(r_axis=85.0)
        result = module.analyze_flip_pair(flip1, flip2, next_row, eye="right")

        assert result["choice"] == "flip2"
        assert result["axis_change"] == pytest.approx(-5.0)
        assert "RAM" in result["intent"]

    def test_no_change_is_both_same(self, module):
        """No axis change (< 1 degree) -> both_same."""
        flip1 = make_row(r_axis=90.0)
        flip2 = make_row(r_axis=90.0)
        next_row = make_row(r_axis=90.0)
        result = module.analyze_flip_pair(flip1, flip2, next_row, eye="right")

        assert result["choice"] == "both_same"

    def test_left_eye_uses_l_axis(self, module):
        """When eye='left', l_axis fields are used."""
        flip1 = make_row(l_axis=90.0)
        flip2 = make_row(l_axis=90.0)
        next_row = make_row(l_axis=95.0)
        result = module.analyze_flip_pair(flip1, flip2, next_row, eye="left")

        assert result["choice"] == "flip1"
        assert result["axis_change"] == pytest.approx(5.0)

    def test_sub_degree_change_is_both_same(self, module):
        """Change of 0.5 degrees (< 1.0) counts as both_same."""
        flip1 = make_row(r_axis=90.0)
        flip2 = make_row(r_axis=90.0)
        next_row = make_row(r_axis=90.5)
        result = module.analyze_flip_pair(flip1, flip2, next_row, eye="right")

        assert result["choice"] == "both_same"


# ── suggest_next_axis ────────────────────────────────────────────────────

class TestSuggestNextAxis:
    """Tests for CylinderAxisModule.suggest_next_axis."""

    def test_flip1_increases(self, module):
        """GAP adds axis_increment (5 degrees)."""
        assert module.suggest_next_axis(90.0, "flip1") == pytest.approx(95.0)

    def test_flip2_decreases(self, module):
        """RAM subtracts axis_increment (5 degrees)."""
        assert module.suggest_next_axis(90.0, "flip2") == pytest.approx(85.0)

    def test_both_same_unchanged(self, module):
        assert module.suggest_next_axis(90.0, "both_same") == pytest.approx(90.0)

    def test_wraparound_high(self, module):
        """178 + 5 = 183 -> wraps to 3 (mod 180)."""
        assert module.suggest_next_axis(178.0, "flip1") == pytest.approx(3.0)

    def test_wraparound_low(self, module):
        """2 - 5 = -3 -> wraps to 177 (mod 180)."""
        assert module.suggest_next_axis(2.0, "flip2") == pytest.approx(177.0)


# ── is_stable ────────────────────────────────────────────────────────────

class TestIsStable:
    """Tests for CylinderAxisModule.is_stable."""

    def test_empty_changes_not_stable(self, module):
        assert module.is_stable([]) is False

    def test_last_change_sub_degree_is_stable(self, module):
        """Last change < 1.0 degree (both same) -> stable."""
        assert module.is_stable([5.0, 0.0]) is True

    def test_oscillation_is_stable(self, module):
        """Positive then negative -> oscillation -> stable."""
        assert module.is_stable([5.0, -5.0]) is True

    def test_consistent_direction_not_stable(self, module):
        assert module.is_stable([5.0, 5.0]) is False

    def test_single_nonzero_change_not_stable(self, module):
        assert module.is_stable([5.0]) is False
