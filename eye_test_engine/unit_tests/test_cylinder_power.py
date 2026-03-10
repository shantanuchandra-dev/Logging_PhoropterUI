"""Unit tests for modules.cylinder_power.CylinderPowerModule."""
import pytest
from conftest import make_row
from eye_test_engine.modules.cylinder_power import CylinderPowerModule


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def module(thresholds):
    return CylinderPowerModule(thresholds)


# ── analyze_flip_pair ────────────────────────────────────────────────────

class TestAnalyzeFlipPair:
    """Tests for CylinderPowerModule.analyze_flip_pair."""

    def test_zero_cylinder_exception(self, module):
        """When cylinder is ~0.00 and Flip1 is chosen -> no_cylinder."""
        flip1 = make_row(r_cyl=0.0, occluder="Right_Flip1_Power")
        flip2 = make_row(r_cyl=0.0, occluder="Right_Flip2_Power")
        result = module.analyze_flip_pair(flip1, flip2, next_row=None, eye="right")

        assert result["choice"] == "no_cylinder"
        assert result["next_cyl"] == 0.0
        assert result["cyl_change"] == 0.0

    def test_no_next_row(self, module):
        """Missing next_row -> choice='unknown'."""
        flip1 = make_row(r_cyl=-1.0, occluder="Right_Flip1_Power")
        flip2 = make_row(r_cyl=-1.0, occluder="Right_Flip2_Power")
        result = module.analyze_flip_pair(flip1, flip2, next_row=None, eye="right")

        assert result["choice"] == "unknown"
        assert result["next_cyl"] == -1.0

    def test_positive_change_is_flip1_gap(self, module):
        """Cyl goes from -1.00 to -0.75 (more positive) -> GAP / flip1."""
        flip1 = make_row(r_cyl=-1.0)
        flip2 = make_row(r_cyl=-1.0)
        next_row = make_row(r_cyl=-0.75)
        result = module.analyze_flip_pair(flip1, flip2, next_row, eye="right")

        assert result["choice"] == "flip1"
        assert result["cyl_change"] == pytest.approx(0.25)
        assert "GAP" in result["intent"]

    def test_negative_change_is_flip2_ram(self, module):
        """Cyl goes from -1.00 to -1.25 (more negative) -> RAM / flip2."""
        flip1 = make_row(r_cyl=-1.0)
        flip2 = make_row(r_cyl=-1.0)
        next_row = make_row(r_cyl=-1.25)
        result = module.analyze_flip_pair(flip1, flip2, next_row, eye="right")

        assert result["choice"] == "flip2"
        assert result["cyl_change"] == pytest.approx(-0.25)
        assert "RAM" in result["intent"]

    def test_no_change_is_both_same(self, module):
        """No change in cyl -> both_same."""
        flip1 = make_row(r_cyl=-1.0)
        flip2 = make_row(r_cyl=-1.0)
        next_row = make_row(r_cyl=-1.0)
        result = module.analyze_flip_pair(flip1, flip2, next_row, eye="right")

        assert result["choice"] == "both_same"
        assert abs(result["cyl_change"]) < 0.01

    def test_left_eye_uses_l_cyl(self, module):
        """When eye='left', l_cyl fields are used."""
        flip1 = make_row(l_cyl=-1.0)
        flip2 = make_row(l_cyl=-1.0)
        next_row = make_row(l_cyl=-0.75)
        result = module.analyze_flip_pair(flip1, flip2, next_row, eye="left")

        assert result["choice"] == "flip1"
        assert result["cyl_change"] == pytest.approx(0.25)


# ── suggest_next_cyl ─────────────────────────────────────────────────────

class TestSuggestNextCyl:
    """Tests for CylinderPowerModule.suggest_next_cyl."""

    def test_flip1_increases(self, module):
        """GAP choice adds power_increment (0.25)."""
        assert module.suggest_next_cyl(-1.0, "flip1") == pytest.approx(-0.75)

    def test_flip2_decreases(self, module):
        """RAM choice subtracts power_increment (0.25)."""
        assert module.suggest_next_cyl(-1.0, "flip2") == pytest.approx(-1.25)

    def test_both_same_unchanged(self, module):
        assert module.suggest_next_cyl(-1.0, "both_same") == pytest.approx(-1.0)

    def test_unknown_unchanged(self, module):
        assert module.suggest_next_cyl(-1.0, "unknown") == pytest.approx(-1.0)


# ── is_stable ────────────────────────────────────────────────────────────

class TestIsStable:
    """Tests for CylinderPowerModule.is_stable."""

    def test_zero_cylinder_is_stable(self, module):
        assert module.is_stable([], current_cyl=0.0) is True

    def test_empty_changes_not_stable(self, module):
        assert module.is_stable([], current_cyl=-1.0) is False

    def test_last_change_zero_is_stable(self, module):
        """'Both Same' = last change ~0 -> stable."""
        assert module.is_stable([0.25, 0.0], current_cyl=-1.0) is True

    def test_oscillation_is_stable(self, module):
        """Positive then negative (or vice versa) = oscillation -> stable."""
        assert module.is_stable([0.25, -0.25], current_cyl=-1.0) is True

    def test_consistent_direction_not_stable(self, module):
        """Same direction changes -> not stable yet."""
        assert module.is_stable([-0.25, -0.25], current_cyl=-1.0) is False

    def test_single_nonzero_change_not_stable(self, module):
        assert module.is_stable([0.25], current_cyl=-1.0) is False
