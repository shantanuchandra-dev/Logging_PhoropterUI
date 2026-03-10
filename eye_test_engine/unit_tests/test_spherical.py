"""Unit tests for modules.spherical.SphericalModule."""
import pytest
from conftest import make_row
from eye_test_engine.modules.spherical import SphericalModule


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def module(thresholds):
    return SphericalModule(thresholds)

<<<<<<< HEAD
# checking unit testing 
=======

>>>>>>> c6614c2 (Adding Unit Test Folder)
# ── analyze_sequence ─────────────────────────────────────────────────────

class TestAnalyzeSequence:
    """Tests for SphericalModule.analyze_sequence."""

    def test_empty_rows(self, module):
        """Empty input should return not-stable with zero counts."""
        result = module.analyze_sequence([], eye="right")

        assert result["is_stable"] is False
        assert result["unable_read_count"] == 0
        assert result["sph_changes"] == []
        assert result["recommendation"] == "continue"

    def test_single_row_no_change(self, module):
        """A single row has no previous row so nothing is tracked."""
        rows = [make_row(r_sph=-1.0, intent="Unable to read")]
        result = module.analyze_sequence(rows, eye="right")

        assert result["unable_read_count"] == 0
        assert result["sph_changes"] == []
        assert result["is_stable"] is False

    def test_two_rows_one_unable(self, module):
        """SPH changes with one 'Unable to read' -> count=1, one_more_attempt."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["unable_read_count"] == 1
        assert result["sph_changes"] == [-0.25]
        assert result["is_stable"] is False
        assert result["recommendation"] == "one_more_attempt"

    def test_threshold_reached_stable(self, module):
        """Two consecutive 'Unable to read' hits threshold=2 -> stable."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Unable to read"),
            make_row(r_sph=-1.50, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["is_stable"] is True
        assert result["unable_read_count"] == 2
        assert result["recommendation"] == "move_to_cylinder"

    def test_able_to_read_resets_count(self, module):
        """'Able to read' after 'Unable' resets the unable counter to 0."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Unable to read"),
            make_row(r_sph=-1.50, intent="Able to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["unable_read_count"] == 0
        assert result["is_stable"] is False
        assert result["recommendation"] == "continue"

    def test_getting_better_resets_count(self, module):
        """'Getting better' also resets the unable counter."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Unable to read"),
            make_row(r_sph=-1.50, intent="Getting better"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["unable_read_count"] == 0
        assert result["is_stable"] is False
  
    def test_left_eye_uses_l_sph(self, module):
        """When eye='left', SPH changes are computed from l_sph."""
        rows = [
            make_row(l_sph=-2.00),
            make_row(l_sph=-2.25, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="left")

        assert result["sph_changes"] == [-0.25]
        assert result["unable_read_count"] == 1

    def test_no_sph_change_skips_row(self, module):
        """Rows where SPH didn't change are not counted."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.00, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["sph_changes"] == []
        assert result["unable_read_count"] == 0

    # ── Manually added tests ─────────────────────────────────────────────

    def test_alternating_never_stable(self, module):
        """Unable→Able→Unable→Able→Unable never hits threshold."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Unable to read"),
            make_row(r_sph=-1.50, intent="Able to read"),
            make_row(r_sph=-1.75, intent="Unable to read"),
            make_row(r_sph=-2.00, intent="Able to read"),
            make_row(r_sph=-2.25, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["is_stable"] is False
        assert result["unable_read_count"] == 1
        assert result["recommendation"] == "one_more_attempt"

    def test_large_sph_jump(self, module):
        """A big SPH change (-1.00) is still tracked as a valid change."""
        rows = [
            make_row(r_sph=-2.00),
            make_row(r_sph=-3.00, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["sph_changes"] == [-1.00]
        assert result["unable_read_count"] == 1


# ── suggest_next_sph ─────────────────────────────────────────────────────

class TestSuggestNextSph:
    """Tests for SphericalModule.suggest_next_sph."""

    def test_unable_to_read(self, module):
        result = module.suggest_next_sph(-1.0, "Unable to read", [])
        assert result == -1.25

    def test_blurry(self, module):
        result = module.suggest_next_sph(-1.0, "Blurry", [])
        assert result == -1.25

    def test_getting_better_negative_history(self, module):
        """Continues in negative direction when last change was negative."""
        result = module.suggest_next_sph(-1.0, "Getting better", [-0.25])
        assert result == -1.25

    def test_getting_better_positive_history(self, module):
        """Goes positive when last change was positive."""
        result = module.suggest_next_sph(-1.0, "Getting better", [0.25])
        assert result == -0.75

    def test_getting_better_empty_history(self, module):
        """Empty history defaults to +0.25."""
        result = module.suggest_next_sph(-1.0, "Getting better", [])
        assert result == -0.75

    def test_unknown_intent_returns_none(self, module):
        result = module.suggest_next_sph(-1.0, "Some random text", [])
        assert result is None

    # ── Manually added test ──────────────────────────────────────────────

    def test_suggest_with_large_negative_sph(self, module):
        """Should work the same even with extreme SPH values."""
        result = module.suggest_next_sph(-10.0, "Unable to read", [])
        assert result == -10.25
