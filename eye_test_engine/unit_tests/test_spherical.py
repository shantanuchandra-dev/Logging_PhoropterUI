"""Unit tests for modules.spherical.SphericalModule."""
import pytest
from conftest import make_row
from eye_test_engine.modules.spherical import SphericalModule


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def module(thresholds):
    return SphericalModule(thresholds)


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

    def test_three_consecutive_unable_exceeds_threshold(self, module):
        """Three consecutive 'Unable to read' still yields stable (threshold=2)."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Unable to read"),
            make_row(r_sph=-1.50, intent="Unable to read"),
            make_row(r_sph=-1.75, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["is_stable"] is True
        assert result["unable_read_count"] == 3
        assert result["recommendation"] == "move_to_cylinder"

    def test_reset_then_re_reach_threshold(self, module):
        """Unable→Able resets, then two more Unable re-triggers stable."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Unable to read"),
            make_row(r_sph=-1.50, intent="Able to read"),
            make_row(r_sph=-1.75, intent="Unable to read"),
            make_row(r_sph=-2.00, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["is_stable"] is True
        assert result["unable_read_count"] == 2
        assert result["recommendation"] == "move_to_cylinder"

    def test_positive_sph_changes(self, module):
        """Positive SPH changes (less minus) are tracked correctly."""
        rows = [
            make_row(r_sph=-3.00),
            make_row(r_sph=-2.75, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["sph_changes"] == [0.25]
        assert result["unable_read_count"] == 1

    def test_all_able_to_read(self, module):
        """Sequence of only 'Able to read' keeps unable_count at 0."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Able to read"),
            make_row(r_sph=-1.50, intent="Able to read"),
            make_row(r_sph=-1.75, intent="Able to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["unable_read_count"] == 0
        assert result["is_stable"] is False
        assert result["recommendation"] == "continue"
        assert len(result["sph_changes"]) == 3

    def test_none_intent_not_counted(self, module):
        """Rows with no intent (None) should not bump the unable counter."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent=None),
            make_row(r_sph=-1.50, intent=None),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["unable_read_count"] == 0
        assert result["is_stable"] is False
        assert len(result["sph_changes"]) == 2

    def test_left_eye_stable(self, module):
        """Full left-eye sequence reaching threshold should become stable."""
        rows = [
            make_row(l_sph=-2.00),
            make_row(l_sph=-2.25, intent="Unable to read"),
            make_row(l_sph=-2.50, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="left")

        assert result["is_stable"] is True
        assert result["unable_read_count"] == 2
        assert result["recommendation"] == "move_to_cylinder"
        assert result["sph_changes"] == [-0.25, -0.25]

    def test_sph_changes_accumulated_correctly(self, module):
        """All SPH deltas are captured in order in sph_changes."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Able to read"),
            make_row(r_sph=-1.50, intent="Getting better"),
            make_row(r_sph=-2.00, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["sph_changes"] == [-0.25, -0.25, -0.50]

    def test_only_right_sph_matters_for_right_eye(self, module):
        """When eye='right', l_sph changes are irrelevant to sph_changes values."""
        rows = [
            make_row(r_sph=-1.00, l_sph=-3.00),
            make_row(r_sph=-1.25, l_sph=-5.00, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["sph_changes"] == [-0.25]

    def test_only_left_sph_matters_for_left_eye(self, module):
        """When eye='left', r_sph changes are irrelevant to sph_changes values."""
        rows = [
            make_row(r_sph=-1.00, l_sph=-2.00),
            make_row(r_sph=-5.00, l_sph=-2.50, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="left")

        assert result["sph_changes"] == [-0.50]

    def test_zero_sph_starting_point(self, module):
        """Sequence starting from SPH 0.0 (plano) works correctly."""
        rows = [
            make_row(r_sph=0.0),
            make_row(r_sph=-0.25, intent="Unable to read"),
            make_row(r_sph=-0.50, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["is_stable"] is True
        assert result["sph_changes"] == [-0.25, -0.25]

    def test_positive_sph_values(self, module):
        """Hyperopic (positive SPH) sequences are handled the same way."""
        rows = [
            make_row(r_sph=2.00),
            make_row(r_sph=2.25, intent="Unable to read"),
            make_row(r_sph=2.50, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["is_stable"] is True
        assert result["sph_changes"] == [0.25, 0.25]
        assert result["recommendation"] == "move_to_cylinder"

    def test_no_change_rows_interleaved(self, module):
        """Rows with no SPH change are silently skipped even between unable rows."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Unable to read"),
            make_row(r_sph=-1.25, intent="Unable to read"),  # no change
            make_row(r_sph=-1.50, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["sph_changes"] == [-0.25, -0.25]
        assert result["unable_read_count"] == 2
        assert result["is_stable"] is True

    def test_long_sequence_stable_at_end(self, module):
        """A long mixed sequence only becomes stable at the very end."""
        rows = [
            make_row(r_sph=-1.00),
            make_row(r_sph=-1.25, intent="Able to read"),
            make_row(r_sph=-1.50, intent="Getting better"),
            make_row(r_sph=-1.75, intent="Able to read"),
            make_row(r_sph=-2.00, intent="Getting better"),
            make_row(r_sph=-2.25, intent="Unable to read"),
            make_row(r_sph=-2.50, intent="Unable to read"),
        ]
        result = module.analyze_sequence(rows, eye="right")

        assert result["is_stable"] is True
        assert result["unable_read_count"] == 2
        assert result["recommendation"] == "move_to_cylinder"
        assert len(result["sph_changes"]) == 6


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

    def test_able_to_read_returns_none(self, module):
        """'Able to read' is not a handled intent for suggest_next_sph."""
        result = module.suggest_next_sph(-1.0, "Able to read", [])
        assert result is None

    def test_suggest_from_zero_sph(self, module):
        """Starting from plano (0.0) should still move by -0.25."""
        result = module.suggest_next_sph(0.0, "Unable to read", [])
        assert result == -0.25

    def test_suggest_from_positive_sph(self, module):
        """Positive (hyperopic) SPH should decrease by 0.25 on 'Unable to read'."""
        result = module.suggest_next_sph(2.0, "Unable to read", [])
        assert result == 1.75

    def test_blurry_from_zero(self, module):
        """'Blurry' at plano goes to -0.25."""
        result = module.suggest_next_sph(0.0, "Blurry", [])
        assert result == -0.25

    def test_getting_better_with_mixed_history(self, module):
        """Only the last entry in history determines direction."""
        result = module.suggest_next_sph(-2.0, "Getting better", [0.25, -0.25, 0.25])
        assert result == -1.75  # last change was +0.25 -> go positive

    def test_getting_better_last_change_negative(self, module):
        """Last history entry negative means continue minus direction."""
        result = module.suggest_next_sph(-2.0, "Getting better", [0.25, 0.25, -0.25])
        assert result == -2.25

    def test_getting_better_single_positive_history(self, module):
        result = module.suggest_next_sph(-3.0, "Getting better", [0.25])
        assert result == -2.75

    def test_empty_string_intent_returns_none(self, module):
        """An empty string intent should return None."""
        result = module.suggest_next_sph(-1.0, "", [])
        assert result is None

    def test_suggest_consecutive_unable(self, module):
        """Calling suggest twice in a row simulates successive adjustments."""
        first = module.suggest_next_sph(-1.0, "Unable to read", [])
        assert first == -1.25
        second = module.suggest_next_sph(first, "Unable to read", [-0.25])
        assert second == -1.50

    def test_blurry_with_positive_sph(self, module):
        """'Blurry' on a positive SPH should still subtract 0.25."""
        result = module.suggest_next_sph(1.0, "Blurry", [])
        assert result == 0.75

    def test_getting_better_history_with_zero(self, module):
        """History ending with 0 (not < 0) takes the positive branch."""
        result = module.suggest_next_sph(-1.0, "Getting better", [0.0])
        assert result == -0.75
