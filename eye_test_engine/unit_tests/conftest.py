"""
Shared fixtures and helpers for module unit tests.

Provides:
- make_row(): factory to build RowContext instances with sensible defaults
- thresholds: pytest fixture mirroring config/thresholds.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pytest

# Add the PARENT of eye_test_engine so the full package is importable.
# This makes `from eye_test_engine.modules.*` work correctly with
# the relative imports inside the module files (e.g. `from ..core.context`).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from eye_test_engine.core.context import RowContext  # noqa: E402


# ---------------------------------------------------------------------------
# RowContext factory
# ---------------------------------------------------------------------------

def make_row(
    *,
    r_sph: float = 0.0,
    r_cyl: float = 0.0,
    r_axis: float = 90.0,
    r_add: float = 0.0,
    l_sph: float = 0.0,
    l_cyl: float = 0.0,
    l_axis: float = 90.0,
    l_add: float = 0.0,
    occluder: str = "BINO",
    chart_display: str = "snellen_chart_200_150",
    chart_number: int = 1,
    intent: Optional[str] = None,
    question: Optional[str] = None,
    timestamp: str = "2025-01-01T00:00:00",
) -> RowContext:
    """Build a RowContext with sensible defaults for testing."""
    return RowContext(
        timestamp=timestamp,
        r_sph=r_sph,
        r_cyl=r_cyl,
        r_axis=r_axis,
        r_add=r_add,
        l_sph=l_sph,
        l_cyl=l_cyl,
        l_axis=l_axis,
        l_add=l_add,
        pd="62",
        chart_number=chart_number,
        occluder_state=occluder,
        chart_display=chart_display,
        ocr_fields_read=0,
        anomalies_fixed=0,
        optometrist_question=question,
        patient_answer_intent=intent,
    )


# ---------------------------------------------------------------------------
# Thresholds fixture (mirrors config/thresholds.yaml)
# ---------------------------------------------------------------------------

@pytest.fixture
def thresholds():
    """Return threshold values matching config/thresholds.yaml."""
    return {
        "sphere_refinement": {
            "unable_read_threshold": 2,
            "sph_change_tolerance": 0.001,
        },
        "cylinder_refinement": {
            "axis_increment": 5,
            "power_increment": 0.25,
            "zero_cylinder_threshold": 0.001,
        },
    }
