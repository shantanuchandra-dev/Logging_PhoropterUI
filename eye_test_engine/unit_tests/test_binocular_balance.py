"""Unit tests for modules.binocular_balance.BinocularBalanceModule."""
import pytest
from conftest import make_row
from eye_test_engine.modules.binocular_balance import BinocularBalanceModule


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def module(thresholds):
    return BinocularBalanceModule(thresholds)


# ── analyze_sequence ─────────────────────────────────────────────────────

class TestAnalyzeSequence:
    """Tests for BinocularBalanceModule.analyze_sequence."""

    def test_empty_rows(self, module):
        """Empty input returns not-balanced with unknown clarity."""
        result = module.analyze_sequence([])

        assert result["is_balanced"] is False
        assert result["clarity_level"] == "unknown"
        assert result["recommendation"] == "No data"

    def test_mostly_able_to_read_is_balanced(self, module):
        """2 out of 3 'Able to read' -> balanced, good clarity."""
        rows = [
            make_row(intent="Able to read"),
            make_row(intent="Able to read"),
            make_row(intent="Blurry"),
        ]
        result = module.analyze_sequence(rows)

        assert result["is_balanced"] is True
        assert result["clarity_level"] == "good"
        assert result["recommendation"] == "Test complete"

    def test_all_able_to_read(self, module):
        """3 out of 3 'Able to read' -> balanced."""
        rows = [
            make_row(intent="Able to read"),
            make_row(intent="Able to read"),
            make_row(intent="Able to read"),
        ]
        result = module.analyze_sequence(rows)
        assert result["is_balanced"] is True

    def test_mostly_unable_is_poor(self, module):
        """2 out of 3 'Unable to read' -> not balanced, poor clarity."""
        rows = [
            make_row(intent="Unable to read"),
            make_row(intent="Unable to read"),
            make_row(intent="Able to read"),
        ]
        result = module.analyze_sequence(rows)

        assert result["is_balanced"] is False
        assert result["clarity_level"] == "poor"
        assert result["recommendation"] == "Re-check refraction"

    def test_mixed_intents_is_uncertain(self, module):
        """Mixed responses with no 2+ majority -> uncertain."""
        rows = [
            make_row(intent="Able to read"),
            make_row(intent="Unable to read"),
            make_row(intent="Blurry"),
        ]
        result = module.analyze_sequence(rows)

        assert result["is_balanced"] is False
        assert result["clarity_level"] == "uncertain"
        assert result["recommendation"] == "Continue testing"

    def test_only_last_three_rows_matter(self, module):
        """Older rows beyond the last 3 should be ignored."""
        rows = [
            make_row(intent="Unable to read"),
            make_row(intent="Unable to read"),
            make_row(intent="Unable to read"),
            make_row(intent="Able to read"),
            make_row(intent="Able to read"),
            make_row(intent="Able to read"),
        ]
        result = module.analyze_sequence(rows)

        assert result["is_balanced"] is True
        assert result["clarity_level"] == "good"

    def test_none_intents_filtered(self, module):
        """Rows with None intent are filtered from the count."""
        rows = [
            make_row(intent=None),
            make_row(intent="Able to read"),
            make_row(intent="Able to read"),
        ]
        result = module.analyze_sequence(rows)
        assert result["is_balanced"] is True

    def test_single_row(self, module):
        """Single 'Able to read' is not enough for balance (need >= 2)."""
        rows = [make_row(intent="Able to read")]
        result = module.analyze_sequence(rows)

        assert result["is_balanced"] is False
        assert result["clarity_level"] == "uncertain"


# ── is_complete ──────────────────────────────────────────────────────────

class TestIsComplete:
    """Tests for BinocularBalanceModule.is_complete."""

    def test_balanced_is_complete(self, module):
        rows = [
            make_row(intent="Able to read"),
            make_row(intent="Able to read"),
            make_row(intent="Able to read"),
        ]
        assert module.is_complete(rows) is True

    def test_unbalanced_is_not_complete(self, module):
        rows = [
            make_row(intent="Unable to read"),
            make_row(intent="Unable to read"),
        ]
        assert module.is_complete(rows) is False

    def test_empty_is_not_complete(self, module):
        assert module.is_complete([]) is False
