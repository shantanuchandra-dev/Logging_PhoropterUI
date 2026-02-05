# JCC Operations Order

## Correct Order of Operations

When adjusting axis or cylinder power during JCC phases, the **JCC operations (`increase`/`decrease`) should be called FIRST**, then the internal state is updated for tracking.

---

## Implementation

### Before (Incorrect)
```python
# OLD - Set power first, then call JCC operation
self.current_row.r_axis += 5
self.set_power(r_axis=self.current_row.r_axis, occluder="Left_Occluded")
self.jcc_flip("increase")  # Called after setting power
```

### After (Correct)
```python
# NEW - Call JCC operation first, then update internal state
self.jcc_flip("increase")  # Phoropter increases axis by 5°

# Update internal state for tracking only
self.current_row = self._copy_row_state()
self.current_row.r_axis += 5
if self.current_row.r_axis > 180:
    self.current_row.r_axis -= 180
```

---

## Rationale

1. **Phoropter Handles the Change**: The JCC `increase`/`decrease` operations tell the phoropter to adjust the value by the configured step size (5° for axis, 0.25D for power)

2. **No Manual Power Setting Needed**: We don't need to call `set_power()` because the JCC operation already adjusts the phoropter

3. **Internal State for Tracking**: We update `self.current_row` values only for internal tracking and display purposes

---

## All Four JCC Phases Updated

### Right Eye Axis
```python
if "GAP Axis" in intent or "Flip 1" in intent:
    self.jcc_flip("increase")  # ← Called FIRST
    
    # Update internal state
    self.current_row = self._copy_row_state()
    self.current_row.r_axis += 5
    if self.current_row.r_axis > 180:
        self.current_row.r_axis -= 180
    
    # Reset to Flip1
    self.jcc_flip("handle")
```

### Left Eye Axis
```python
if "GAP Axis" in intent or "Flip 1" in intent:
    self.jcc_flip("increase")  # ← Called FIRST
    
    # Update internal state
    self.current_row = self._copy_row_state()
    self.current_row.l_axis += 5
    if self.current_row.l_axis > 180:
        self.current_row.l_axis -= 180
    
    # Reset to Flip1
    self.jcc_flip("handle")
```

### Right Eye Power
```python
if "GAP Power" in intent or "Flip 1" in intent:
    self.jcc_flip("increase")  # ← Called FIRST
    
    # Update internal state
    self.current_row = self._copy_row_state()
    self.current_row.r_cyl += 0.25
    
    # Reset to Flip1
    self.jcc_flip("handle")
```

### Left Eye Power
```python
if "GAP Power" in intent or "Flip 1" in intent:
    self.jcc_flip("increase")  # ← Called FIRST
    
    # Update internal state
    self.current_row = self._copy_row_state()
    self.current_row.l_cyl += 0.25
    
    # Reset to Flip1
    self.jcc_flip("handle")
```

---

## API Call Sequence

### Example: Right Eye Axis Adjustment (Increase)

**Correct Order:**
```bash
# 1. JCC increase operation (phoropter adjusts axis)
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "increase"}]}'

# 2. JCC handle operation (reset to Flip1)
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "handle"}]}'

# No set_power() call needed!
```

**Old (Incorrect) Order:**
```bash
# 1. Set power (manual adjustment)
curl -X POST .../run-tests \
  -d '{"test_cases": [{"right_eye": {"axis": 95}, "aux_lens": "AuxLensL"}]}'

# 2. JCC increase operation (redundant)
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "increase"}]}'

# 3. JCC handle operation
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "handle"}]}'
```

---

## Console Output

### Before (with set_power)
```
✓ Power set: R(None/None/95) L(None/None/None) Occ: Left_Occluded
✓ JCC eye mode set: L
✓ JCC action: increase
✓ JCC action: handle
```

### After (without set_power)
```
✓ JCC action: increase
✓ JCC action: handle
```

Much cleaner! The phoropter handles the value adjustment internally.

---

## Benefits

1. **Cleaner API Calls**: Fewer redundant calls to the phoropter
2. **Correct Operation Order**: JCC operations are called in the right sequence
3. **Phoropter Controls Values**: The phoropter manages its own state
4. **Internal Tracking**: Our code tracks values for display/logging only
5. **Consistent with JCC Behavior**: Follows the intended JCC workflow

---

## Files Modified

- **`eye_test_engine/interactive_session.py`**
  - Updated `_process_jcc_axis_right()` - Removed `set_power()` calls
  - Updated `_process_jcc_axis_left()` - Removed `set_power()` calls
  - Updated `_process_jcc_power_right()` - Removed `set_power()` calls
  - Updated `_process_jcc_power_left()` - Removed `set_power()` calls

---

## Summary

| Operation | Order | Purpose |
|-----------|-------|---------|
| `jcc_flip("increase")` | **1st** | Phoropter adjusts value (+5° or +0.25D) |
| Update `self.current_row` | **2nd** | Internal tracking for display |
| `jcc_flip("handle")` | **3rd** | Reset to Flip1 for next cycle |

**Key Point**: The phoropter handles the actual value changes. We only update our internal state for tracking purposes.

---

## Date
February 5, 2026

## Status
✅ Fixed - JCC operations now called in correct order (increase/decrease FIRST, then internal state update)
