# JCC Chart Behavior - Clarification

## Key Understanding

**After calling `set_chart("jcc_chart")`, the JCC chart automatically defaults to Flip 1 of Axis mode.**

No additional API calls are needed to initialize the JCC state.

---

## What NOT to Call After JCC Chart

### ❌ Don't Call: `jcc_flip("R")` or `jcc_flip("L")`
After displaying the JCC chart, do NOT call eye mode APIs:
```python
# WRONG - Don't do this
self.set_chart("jcc_chart")
self.jcc_flip("R")  # ❌ Not needed!
```

```python
# CORRECT
self.set_chart("jcc_chart")
# Chart is ready - defaults to Flip 1 of Axis
```

### ❌ Don't Call: `aux_lens` OFF
After displaying the JCC chart, do NOT set aux_lens to OFF:
```python
# WRONG - Don't do this
self.set_chart("jcc_chart")
self.set_power(occluder="BINO")  # ❌ Sets aux_lens OFF - not needed!
```

---

## JCC Chart Default State

When `jcc_chart` is displayed:
- **Flip Position**: Flip 1 (automatically)
- **Mode**: Axis mode (automatically)
- **Ready to use**: Yes - no initialization needed

---

## What API Calls ARE Needed

### ✅ Flip Between Positions (handle)
To toggle between Flip 1 and Flip 2:
```python
self.jcc_flip("handle")  # Flips from position 1 to 2 (or 2 to 1)
```

### ✅ Switch Between Axis and Power
To switch from Axis mode to Power mode:
```python
self.jcc_flip("power_axis_switch")  # Switches mode
```

### ✅ Adjust Values
To increase or decrease the current value:
```python
self.jcc_flip("increase")  # Increase axis or power
self.jcc_flip("decrease")  # Decrease axis or power
```

---

## Correct JCC Flow

### Initial JCC Axis (Right Eye)
```python
# Step 1: Display JCC chart
self.set_chart("jcc_chart")
# ✓ Chart is now showing Flip 1 of Axis mode

# Step 2: Wait 2 seconds (frontend countdown)

# Step 3: Flip to position 2
self.jcc_flip("handle")
# ✓ Now showing Flip 2

# Step 4: Patient selects Flip 1 or Flip 2

# Step 5: Adjust axis
self.jcc_flip("increase")  # or "decrease"

# Step 6: Reset to Flip 1
self.jcc_flip("handle")
# ✓ Back to Flip 1, ready for next cycle
```

### Transition to Power Mode
```python
# From Axis mode to Power mode
self.jcc_flip("power_axis_switch")
# ✓ Now in Power mode, showing Flip 1

# Continue with same flip cycle as Axis
```

---

## Changes Made

### Removed from `_transition_to_jcc_axis_right()`:
```python
# BEFORE
self.set_chart("jcc_chart")
self.jcc_flip("R")  # ❌ Removed

# AFTER
self.set_chart("jcc_chart")
# Chart is ready - no additional calls needed
```

### Removed from `_transition_to_jcc_axis_left()`:
```python
# BEFORE
self.jcc_flip("L")  # ❌ Removed

# AFTER
# No JCC API calls needed - chart maintains state
```

### Removed from `set_power()`:
```python
# BEFORE
if jcc_eye_mode:
    self.jcc_flip(jcc_eye_mode)  # ❌ Removed

# AFTER
# JCC eye mode is NOT set via set_power()
# Chart defaults to correct state when displayed
```

---

## Aux Lens Usage

### ✅ Still Used for Non-JCC Phases
Aux lens (AuxLensL/AuxLensR) is still used during regular refraction:

```python
# Right eye refraction (Left occluded)
self.set_power(r_sph=-0.25, occluder="Left_Occluded")
# Sets aux_lens to AuxLensL ✓

# Left eye refraction (Right occluded)
self.set_power(l_sph=-0.25, occluder="Right_Occluded")
# Sets aux_lens to AuxLensR ✓
```

### ❌ NOT Used After JCC Chart
After JCC chart is displayed, don't call `set_power()` with occluder:

```python
# WRONG
self.set_chart("jcc_chart")
self.set_power(occluder="Left_Occluded")  # ❌ Not needed

# CORRECT
self.set_chart("jcc_chart")
# Chart handles its own state
```

---

## Summary

| Action | When | API Call | Needed? |
|--------|------|----------|---------|
| Display JCC chart | Initial | `set_chart("jcc_chart")` | ✅ Yes |
| Set eye mode | After chart | `jcc_flip("R")` or `jcc_flip("L")` | ❌ No |
| Set aux lens OFF | After chart | `aux_lens: "OFF"` | ❌ No |
| Flip to position 2 | During test | `jcc_flip("handle")` | ✅ Yes |
| Switch to power | Axis → Power | `jcc_flip("power_axis_switch")` | ✅ Yes |
| Adjust value | After selection | `jcc_flip("increase/decrease")` | ✅ Yes |
| Reset to Flip 1 | After adjustment | `jcc_flip("handle")` | ✅ Yes |

---

## Files Modified

- **`eye_test_engine/interactive_session.py`**
  - Removed `jcc_flip("R")` from `_transition_to_jcc_axis_right()`
  - Removed `jcc_flip("L")` from `_transition_to_jcc_axis_left()`
  - Removed automatic `jcc_flip()` calls from `set_power()`
  - Removed `jcc_flip("R")` and `jcc_flip("L")` from `_setup_phase()`
  - Kept `jcc_flip("power_axis_switch")` for mode switching

## Date
February 5, 2026

## Status
✅ Fixed - JCC chart initialization simplified, unnecessary API calls removed
