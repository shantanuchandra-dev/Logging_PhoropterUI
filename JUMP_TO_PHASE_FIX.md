# Jump to Phase Feature - Fixes and Improvements

## Overview
Fixed the "Jump to Phase" feature to ensure correct charts, sequences, and state initialization when jumping to any phase in the eye test workflow.

## Changes Made

### 1. **interactive_session.py - `_setup_phase()` method** (Lines ~1485-1619)

#### Issues Fixed:
1. **Wrong starting charts for refraction phases**
   - Previously started with `snellen_chart_20_20_20` (middle of sequence)
   - Now starts with `self.snellen_charts[0]` (first/largest chart in sequence)

2. **Undefined method calls**
   - Fixed `self.jcc_flip()` → `self.jcc_control()` (lines 1464, 1486 in old code)

3. **Missing JCC chart display**
   - Added explicit `self.set_chart("jcc_chart")` for all JCC phases
   - Previously relied on implicit chart display

4. **Missing JCC eye mode setup**
   - Added `self.jcc_control("R")` for right eye JCC phases
   - Added `self.jcc_control("L")` for left eye JCC phases

5. **Missing state tracking resets**
   - Added `self._reset_jcc_choice_tracking()` for JCC phases
   - Added `self._reset_duochrome_choice_tracking()` for duochrome phases
   - Added `self.jcc_power_zero_flip1_count = 0` for JCC power phases

6. **Missing auto_flip flag for JCC phases**
   - Method now returns a response dict instead of void
   - JCC phases return response with `auto_flip: True` and `flip_wait_seconds: 2`
   - Non-JCC phases return standard response

#### Detailed Changes by Phase:

**distance_vision:**
- ✅ Correct chart: `echart_400`
- ✅ Correct occluder: `BINO`

**right_eye_refraction:**
- ✅ Fixed: Now starts with `snellen_charts[0]` (was `snellen_chart_20_20_20`)
- ✅ Resets `current_chart_index = 0`
- ✅ Correct occluder: `Left_Occluded`

**jcc_axis_right:**
- ✅ Added: `_reset_jcc_choice_tracking()`
- ✅ Added: `set_chart("jcc_chart")`
- ✅ Added: `jcc_control("R")` for eye mode
- ✅ Added: Returns response with `auto_flip: True`

**jcc_power_right:**
- ✅ Added: `_reset_jcc_choice_tracking()`
- ✅ Added: `jcc_power_zero_flip1_count = 0`
- ✅ Added: `set_chart("jcc_chart")`
- ✅ Added: `jcc_control("R")` for eye mode
- ✅ Fixed: `jcc_control("power_axis_switch")` (was `jcc_flip()`)
- ✅ Added: Returns response with `auto_flip: True`

**duochrome_right:**
- ✅ Added: `_reset_duochrome_choice_tracking()`
- ✅ Correct chart: `duochrome`
- ✅ Correct occluder: `Left_Occluded`

**left_eye_refraction:**
- ✅ Fixed: Now starts with `snellen_charts[0]` (was `snellen_chart_20_20_20`)
- ✅ Resets `current_chart_index = 0`
- ✅ Correct occluder: `Right_Occluded`

**jcc_axis_left:**
- ✅ Added: `_reset_jcc_choice_tracking()`
- ✅ Added: `set_chart("jcc_chart")`
- ✅ Added: `jcc_control("L")` for eye mode
- ✅ Added: Returns response with `auto_flip: True`

**jcc_power_left:**
- ✅ Added: `_reset_jcc_choice_tracking()`
- ✅ Added: `jcc_power_zero_flip1_count = 0`
- ✅ Added: `set_chart("jcc_chart")`
- ✅ Added: `jcc_control("L")` for eye mode
- ✅ Fixed: `jcc_control("power_axis_switch")` (was `jcc_flip()`)
- ✅ Added: Returns response with `auto_flip: True`

**duochrome_left:**
- ✅ Added: `_reset_duochrome_choice_tracking()`
- ✅ Correct chart: `duochrome`
- ✅ Correct occluder: `Right_Occluded`

**binocular_balance:**
- ✅ Correct chart: `snellen_chart_20_20_20`
- ✅ Correct occluder: `BINO`
- ✅ Added: `jcc_control("BINO")`

### 2. **api_server.py - `jump_to_phase()` endpoint** (Lines 96-121)

#### Changes:
- Updated to use the response dict returned by `_setup_phase()`
- Previously called `_setup_phase()` then `_build_response()` separately
- Now uses single response from `_setup_phase()` which includes auto_flip flag

**Before:**
```python
session._setup_phase(target_phase)
state = session._build_response()
```

**After:**
```python
# _setup_phase now returns a response dict with all necessary state
state = session._setup_phase(target_phase)
```

## Testing Checklist

When testing the "Jump to Phase" feature, verify:

### Refraction Phases (Right & Left)
- [ ] Starts with the **first chart** in sequence (`snellen_chart_200_150`)
- [ ] Chart selector shows correct current chart
- [ ] Can progress through chart sequence normally
- [ ] Correct occluder state (Left_Occluded for right eye, Right_Occluded for left eye)

### JCC Phases (Axis & Power, Right & Left)
- [ ] JCC chart is displayed
- [ ] Shows Flip 1 initially
- [ ] Auto-flips to Flip 2 after 2 seconds
- [ ] Correct eye mode (R for right eye phases, L for left eye phases)
- [ ] Choice tracking is reset (no false reversals from previous session)
- [ ] Power mode switch works correctly (for power phases)

### Duochrome Phases (Right & Left)
- [ ] Duochrome chart is displayed
- [ ] Correct occluder state
- [ ] Choice tracking is reset (no false reversals from previous session)

### Distance Vision
- [ ] E-chart 400 is displayed
- [ ] Binocular mode (BINO)

### Binocular Balance
- [ ] Snellen 20/20/20 chart is displayed
- [ ] Binocular mode (BINO)
- [ ] JCC mode set to BINO

## Impact

These fixes ensure that:
1. **Correct starting point**: Each phase starts with the appropriate chart in the sequence
2. **Proper state initialization**: All tracking variables are reset appropriately
3. **Consistent behavior**: Jumping to a phase behaves the same as reaching it naturally
4. **Auto-flip support**: JCC phases automatically flip after 2 seconds when jumped to
5. **No false triggers**: Choice tracking is reset to prevent false reversal detection

## Related Files
- `eye_test_engine/interactive_session.py` - Main session logic
- `eye_test_engine/api_server.py` - API endpoint handler
- `eye_test_engine/frontend/app.js` - Frontend jump to phase implementation
- `eye_test_engine/docs/PHASE_JUMP_FEATURE.md` - Feature documentation
