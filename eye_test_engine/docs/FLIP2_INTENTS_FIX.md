# Fix: Missing Flip 2 Intents

## Problem

Intent buttons were not appearing after the automatic transition from Flip 1 to Flip 2 in JCC phases.

## Root Cause

The `RowContext` class has derived fields (`is_flip1`, `is_flip2`, etc.) that are calculated in `__post_init__()` based on the `occluder_state` string:

```python
def _derive_flip_state(self):
    """Determine flip state from occluder."""
    occ = self.occluder_state.strip()
    
    self.is_flip1 = "Flip1" in occ
    self.is_flip2 = "Flip2" in occ
```

These derived fields are only calculated **once** during object initialization. When we manually changed `occluder_state` later (e.g., `self.current_row.occluder_state = "Right_Axis_Flip2"`), the derived boolean flags (`is_flip1`, `is_flip2`) were **not recalculated**.

The `get_intents()` method relies on these flags:

```python
def get_intents(self) -> List[str]:
    if self.current_row.is_flip1:
        return []  # No intents for Flip1
    elif self.current_row.is_flip2:
        return flip2_intents  # Return Flip2 intents
```

Since `is_flip2` was still `False` after changing `occluder_state` to `"Right_Axis_Flip2"`, the method returned an empty intent list instead of the Flip 2 intents.

## Solution

### 1. Added `update_derived_fields()` method to `RowContext`

```python
def update_derived_fields(self):
    """Recalculate all derived fields after manual state changes."""
    self._derive_chart_type()
    self._derive_flip_state()
    self._derive_eye_tested()
```

This method allows us to manually trigger recalculation of derived fields.

### 2. Added `_update_state()` helper method to `InteractiveSession`

```python
def _update_state(self, occluder: str = None, chart: str = None):
    """Update occluder and/or chart state and refresh derived fields."""
    if occluder is not None:
        self.current_row.occluder_state = occluder
    if chart is not None:
        self.current_row.chart_display = chart
    # Recalculate derived fields after manual changes
    self.current_row.update_derived_fields()
```

This helper ensures that whenever we change `occluder_state` or `chart_display`, the derived fields are automatically recalculated.

### 3. Replaced manual state assignments with `_update_state()` calls

**Before:**
```python
self.current_row.occluder_state = "Right_Axis_Flip2"
```

**After:**
```python
self._update_state(occluder="Right_Axis_Flip2")
```

This was applied to all JCC flip state transitions:
- `Right_Axis_Flip1` ↔ `Right_Axis_Flip2`
- `Left_Axis_Flip1` ↔ `Left_Axis_Flip2`
- `Right_Power_Flip1` ↔ `Right_Power_Flip2`
- `Left_Power_Flip1` ↔ `Left_Power_Flip2`

## Files Modified

1. **`eye_test_engine/core/context.py`**
   - Added `update_derived_fields()` method

2. **`eye_test_engine/interactive_session.py`**
   - Added `_update_state()` helper method
   - Replaced all JCC flip state assignments with `_update_state()` calls
   - Updated all 4 "Repeat" handlers in JCC phases to use `_update_state()`

## Testing

After this fix:
1. ✅ Flip 1 shows with no intent buttons (waiting message)
2. ✅ After 2 seconds, Flip 2 shows automatically
3. ✅ **Intent buttons appear correctly for Flip 2:**
   - "Flip 1 was better (GAP Axis...)"
   - "Flip 2 was better (RAM Axis...)"
   - "Both Same"
   - "Repeat"

## Why This Works

Now when we transition from Flip1 to Flip2:
1. `self._update_state(occluder="Right_Axis_Flip2")` is called
2. This sets `self.current_row.occluder_state = "Right_Axis_Flip2"`
3. **AND** calls `self.current_row.update_derived_fields()`
4. Which recalculates `is_flip2 = True`
5. So `get_intents()` correctly returns the Flip 2 intent list
6. Frontend displays the intent buttons ✅

## Additional Verification

### Axis → Power Transition

The transition from axis refinement to power refinement was verified to be working correctly:

```python
elif "Both Same" in intent or "Reverse" in intent:
    # Move to JCC Power
    return self._transition_to_jcc_power_right()
```

When the patient selects "Both Same" during axis refinement, the system correctly transitions to power refinement. No code changes were needed for this - it was already implemented correctly.
