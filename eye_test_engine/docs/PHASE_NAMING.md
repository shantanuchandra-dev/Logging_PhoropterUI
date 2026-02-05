# Phase Naming Convention

## Overview

All phases now display with proper phase letters (A, B, D, E, F, G, H, I, J, K) and step numbers.

---

## Phase Mapping

| Phase ID | Phase Letter | Phase Name | Clinical Step |
|----------|--------------|------------|---------------|
| `distance_vision` | **A** | Distance Vision | Step 2.1 |
| `right_eye_refraction` | **B** | Right Eye Refraction | Step 6.1 |
| `jcc_axis_right` | **E** | JCC Axis Right | Step 6.2 |
| `jcc_power_right` | **F** | JCC Power Right | Step 6.2 |
| `duochrome_right` | **G** | Duochrome Right | Step 6.2 |
| `left_eye_refraction` | **D** | Left Eye Refraction | Step 6.3 |
| `jcc_axis_left` | **H** | JCC Axis Left | Step 6.4 |
| `jcc_power_left` | **I** | JCC Power Left | Step 6.4 |
| `duochrome_left` | **J** | Duochrome Left | Step 6.4 |
| `binocular_balance` | **K** | Binocular Balance | Step 6.5 |

---

## Implementation

### Phase Name Dictionary

Located in `__init__()` of `InteractiveSession`:

```python
self.phase_names = {
    "distance_vision": "Phase A: Distance Vision (Step 2.1)",
    "right_eye_refraction": "Phase B: Right Eye Refraction (Step 6.1)",
    "jcc_axis_right": "Phase E: JCC Axis Right (Step 6.2)",
    "jcc_power_right": "Phase F: JCC Power Right (Step 6.2)",
    "duochrome_right": "Phase G: Duochrome Right (Step 6.2)",
    "left_eye_refraction": "Phase D: Left Eye Refraction (Step 6.3)",
    "jcc_axis_left": "Phase H: JCC Axis Left (Step 6.4)",
    "jcc_power_left": "Phase I: JCC Power Left (Step 6.4)",
    "duochrome_left": "Phase J: Duochrome Left (Step 6.4)",
    "binocular_balance": "Phase K: Binocular Balance (Step 6.5)",
}
```

### Usage in `_build_response()`

```python
def _build_response(self) -> Dict:
    """Build response with current state."""
    question = self.get_question()
    intents = self.get_intents()
    
    # Get formatted phase name with letter (A, B, etc.)
    phase_display = self.phase_names.get(self.current_phase, self.current_phase)
    
    return {
        "phase": phase_display,  # Returns "Phase A: Distance Vision (Step 2.1)"
        "question": question,
        "intents": intents,
        ...
    }
```

### Console Output

All transition methods now print the phase name:

```python
def _transition_to_jcc_axis_right(self) -> Dict:
    """Transition to JCC axis refinement for right eye."""
    self.current_phase = "jcc_axis_right"
    print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
    ...
```

---

## Console Output Examples

### Starting the Test
```
============================================================
PHASE A: DISTANCE VISION (STEP 2.1)
============================================================
✓ Displaying: echart_400
✓ Power set: R(None/None/None) L(None/None/None) Occ: BINO
✓ JCC eye mode set: BINO
```

### Phase Transitions
```
→ Transitioning to Phase B: Right Eye Refraction (Step 6.1)
✓ Displaying: snellen_chart_200_150
✓ Power set: R(None/None/None) L(None/None/None) Occ: Left_Occluded
✓ JCC eye mode set: L

→ Transitioning to Phase E: JCC Axis Right (Step 6.2)
✓ Displaying: jcc_chart

→ Transitioning to Phase F: JCC Power Right (Step 6.2)
✓ JCC action: power_axis_switch

→ Transitioning to Phase G: Duochrome Right (Step 6.2)
✓ Displaying: duochrome
✓ Power set: R(None/None/None) L(None/None/None) Occ: Left_Occluded
✓ JCC eye mode set: L

→ Transitioning to Phase D: Left Eye Refraction (Step 6.3)
✓ Displaying: snellen_chart_200_150
✓ Power set: R(None/None/None) L(None/None/None) Occ: Right_Occluded
✓ JCC eye mode set: R

→ Transitioning to Phase H: JCC Axis Left (Step 6.4)
✓ Displaying: jcc_chart

→ Transitioning to Phase I: JCC Power Left (Step 6.4)
✓ JCC action: power_axis_switch

→ Transitioning to Phase J: Duochrome Left (Step 6.4)
✓ Displaying: duochrome
✓ Power set: R(None/None/None) L(None/None/None) Occ: Right_Occluded
✓ JCC eye mode set: R

→ Transitioning to Phase K: Binocular Balance (Step 6.5)
✓ Displaying: snellen_chart_20_20_20
✓ Power set: R(None/None/None) L(None/None/None) Occ: BINO
✓ JCC eye mode set: BINO
```

---

## Frontend Display

The frontend receives the formatted phase name in the API response:

```json
{
  "phase": "Phase E: JCC Axis Right (Step 6.2)",
  "question": "Focus on the dot chart. This is Flip 1...",
  "intents": [],
  "chart": "jcc_chart",
  "occluder": "Right_Axis_Flip1",
  "power": { ... }
}
```

The phase badge in the UI displays: **"Phase E: JCC Axis Right (Step 6.2)"**

---

## Benefits

1. **Clear Phase Identification**: Each phase has a unique letter (A-K)
2. **Clinical Context**: Step numbers map to clinical protocol
3. **Consistent Naming**: Same format everywhere (console, API, frontend)
4. **Easy Debugging**: Console output clearly shows phase transitions
5. **User-Friendly**: Frontend displays meaningful phase names

---

## Files Modified

- **`eye_test_engine/interactive_session.py`**
  - Added `self.phase_names` dictionary in `__init__()`
  - Updated `_build_response()` to use phase names
  - Updated `start_distance_vision()` to use phase names
  - Added print statements to all `_transition_to_*()` methods
  - Added print statement to `_process_distance_vision()`

---

## Phase Flow Visualization

```
Phase A: Distance Vision (Step 2.1)
    ↓
Phase B: Right Eye Refraction (Step 6.1)
    ↓
Phase E: JCC Axis Right (Step 6.2)
    ↓
Phase F: JCC Power Right (Step 6.2)
    ↓
Phase G: Duochrome Right (Step 6.2)
    ↓
Phase D: Left Eye Refraction (Step 6.3)
    ↓
Phase H: JCC Axis Left (Step 6.4)
    ↓
Phase I: JCC Power Left (Step 6.4)
    ↓
Phase J: Duochrome Left (Step 6.4)
    ↓
Phase K: Binocular Balance (Step 6.5)
    ↓
Test Complete
```

---

## Date
February 5, 2026

## Status
✅ Complete - All phases display with proper letters and step numbers
