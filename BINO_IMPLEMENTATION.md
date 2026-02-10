# BINO Binocular Balance Implementation

## Overview

This document describes the implementation of the BINO (Binocular Balance) phase in the eye test engine. The BINO phase is the final phase of the refraction test where both eyes are tested together to ensure they are balanced.

## Phase Details

- **Phase ID**: K
- **Phase Name**: Binocular Balance (Step 6.5)
- **Chart**: chart_20 (bino_chart)
- **Occluder State**: BINO (both eyes open)

## Question

```
"You should see 2 lines at top and bottom.
Focus on last letter. Which one is less blurry than the others (if there is one)?"
```

## Intent Logic

The BINO phase presents the patient with two lines (top and bottom) and asks which one is less blurry. The logic is:

### 1. Top is blurry [Right Eye]
- **Action**: Add 0.25D Sph in Left Eye
- **Rationale**: If the top line (viewed by the right eye) is blurry, we compensate by adding plus power to the left eye to balance the binocular vision
- **Implementation**:
  ```python
  self.current_row.l_sph += 0.25
  ```

### 2. Bottom is blurry [Left Eye]
- **Action**: Add 0.25D Sph in Right Eye
- **Rationale**: If the bottom line (viewed by the left eye) is blurry, we compensate by adding plus power to the right eye to balance the binocular vision
- **Implementation**:
  ```python
  self.current_row.r_sph += 0.25
  ```

### 3. Both are same
- **Action**: Test complete
- **Rationale**: When both lines appear equally clear, the binocular balance is achieved and the test is complete
- **Implementation**:
  ```python
  return {
      "phase": "complete",
      "status": "complete",
      "question": "Test complete!",
      "intents": [],
  }
  ```

### 4. Prev State (available after power adjustment)
- **Action**: Restore previous power state
- **Rationale**: If the patient indicates the adjustment made things worse, we can revert to the previous state
- **Implementation**:
  ```python
  self.current_row = self._copy_row_from_dict(self.previous_state)
  self.set_power(r_sph=..., l_sph=..., occluder="BINO")
  ```

## Iterative Process

The BINO phase is designed to be iterative:

1. Show chart_20 with both eyes open
2. Ask the question
3. Patient responds with which line is blurry (or both are same)
4. Adjust power accordingly
5. Repeat steps 2-4 until "Both are same" is selected

### Example Flow

```
Initial State: R(-1.00/-0.50/90) L(-1.50/-0.50/85)

Round 1:
  Q: Which line is less blurry?
  A: Bottom is blurry [Left Eye]
  → Add 0.25D to Right Eye SPH
  New State: R(-0.75/-0.50/90) L(-1.50/-0.50/85)

Round 2:
  Q: Which line is less blurry?
  A: Bottom is blurry [Left Eye]
  → Add 0.25D to Right Eye SPH
  New State: R(-0.50/-0.50/90) L(-1.50/-0.50/85)

Round 3:
  Q: Which line is less blurry?
  A: Both are same
  → Test Complete!
  Final State: R(-0.50/-0.50/90) L(-1.50/-0.50/85)
```

## Implementation Details

### Chart Mapping

Added `chart_20` to the chart mapping:

```python
self.chart_map = {
    # ... existing charts ...
    "bino_chart": "chart_20",
}
```

### Transition Function

Updated `_transition_to_binocular_balance()` to:
- Display chart_20 (bino_chart)
- Set occluder to BINO
- Reset previous state tracking
- Set JCC control to BINO mode

```python
def _transition_to_binocular_balance(self) -> Dict:
    self.current_phase = "binocular_balance"
    self.previous_state = None
    self.show_prev_state_option = False
    
    self.current_row = self._copy_row_state()
    self.current_row.occluder_state = "BINO"
    self.current_row.chart_display = "bino_chart"
    
    self.set_chart("bino_chart")
    self.set_power(occluder="BINO")
    self.jcc_control("BINO")
    
    return self._build_response()
```

### Process Function

Implemented `_process_binocular_balance()` with full logic for:
- Top is blurry → Add 0.25D to Left Eye
- Bottom is blurry → Add 0.25D to Right Eye
- Both are same → Test complete
- Prev State → Restore previous power

### Protocol Configuration

Updated `protocol.yaml` to reflect the new question and intents:

```yaml
binocular_balance:
  id: "K"
  name: "Binocular Balance (Step 6.5)"
  trigger:
    occluder: "BINO"
    charts: ["bino_chart"]
    condition: "after_first_duochrome"
  questions:
    - "You should see 2 lines at top and bottom. Focus on last letter. Which one is less blurry than the others (if there is one)?"
  intents:
    - "Top is blurry [Right Eye]"
    - "Bottom is blurry [Left Eye]"
    - "Both are same"
  adjustment_rules:
    top_blurry: "+0.25D to L_SPH"
    bottom_blurry: "+0.25D to R_SPH"
```

## API Usage

To display the BINO chart (chart_20), use the following curl command:

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_20"] } }] }'
```

## Testing

Comprehensive tests have been implemented in `tests/test_binocular_balance_logic.py`:

1. **test_binocular_balance_top_blurry**: Verifies that selecting "Top is blurry" adds 0.25D to Left Eye SPH
2. **test_binocular_balance_bottom_blurry**: Verifies that selecting "Bottom is blurry" adds 0.25D to Right Eye SPH
3. **test_binocular_balance_both_same**: Verifies that selecting "Both are same" completes the test
4. **test_binocular_balance_prev_state**: Verifies that "Prev State" restores the previous power
5. **test_binocular_balance_iterative**: Verifies multiple rounds of adjustments until balanced

All tests pass successfully:

```
============================================================
ALL TESTS PASSED ✓
============================================================
```

## Previous State Tracking

The BINO phase uses the same "Prev State" mechanism as the refraction phases:

1. Before making any power adjustment, the current state is saved to `self.previous_state`
2. After adjustment, "Prev State" option is added to the intents list
3. If patient selects "Prev State", the previous power is restored
4. After restoring, "Prev State" option is removed from intents

This allows the patient to undo an adjustment if it made things worse.

## Integration with Existing Phases

The BINO phase is triggered after the left eye duochrome test completes:

```
Phase Flow:
... → Duochrome Left (J) → Binocular Balance (K) → Test Complete
```

The transition is handled in `_process_duochrome_left()`:

```python
elif intent == "Both Same":
    return self._transition_to_binocular_balance()
```

## Jump to Phase Support

The BINO phase is fully integrated with the "Jump to Phase" feature. When jumping to BINO:

1. Chart is set to bino_chart (chart_20)
2. Occluder is set to BINO
3. Previous state tracking is reset
4. JCC control is set to BINO mode

## Summary

The BINO binocular balance phase has been fully implemented with:

✅ Chart_20 (bino_chart) display
✅ Custom question for binocular balance
✅ Intent logic for top/bottom blurry and both same
✅ Power adjustment logic (+0.25D to opposite eye)
✅ Iterative process until balanced
✅ Previous state tracking and restoration
✅ Test completion when balanced
✅ Full integration with existing phases
✅ Jump to Phase support
✅ Comprehensive test coverage

The implementation follows the same patterns and conventions as the existing phases, ensuring consistency and maintainability.
