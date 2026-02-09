# Prev State Feature - Implementation Summary

## Overview
Implemented a "Prev State" option that appears when a user clicks "Blurry" during eye refraction tests (covering eye tests). This allows the user to revert to the previous power settings if they want to try again.

## Changes Made

### 1. Modified `interactive_session.py`

#### Added State Tracking (Lines 44-48)
```python
# Previous state tracking for "Prev State" functionality
self.previous_state = None
self.show_prev_state_option = False
```

#### Updated `get_intents()` Method (Lines 256-275)
- Modified to append "Prev State" to the intents list when:
  - `show_prev_state_option` is True
  - `previous_state` is not None
  - Current phase is a refraction phase (not JCC phase)

#### Updated `_process_right_eye_refraction()` Method (Lines 360-415)
Added the following logic:

1. **Handle "Prev State" Intent**: Restores the previous power settings and removes the "Prev State" option
2. **Save State Before "Blurry"**: When user clicks "Blurry", saves current power settings before making -0.25D adjustment
3. **Enable "Prev State" Option**: Sets `show_prev_state_option = True` after "Blurry" response
4. **Reset Option for Other Intents**: Clears `show_prev_state_option` when user selects any other intent

#### Updated `_process_left_eye_refraction()` Method (Lines 417-472)
Applied identical logic as right eye refraction for consistency.

#### Added `_copy_row_from_dict()` Helper Method (Lines 869-880)
New helper method to restore state from a saved dictionary.

## Behavior

### When User Clicks "Blurry"
1. System saves current power settings (SPH, CYL, AXIS for both eyes)
2. Applies -0.25D SPH adjustment as usual
3. Sends CURL command to phoropter with new power
4. Returns response with "Prev State" added to intents list

### When User Clicks "Prev State"
1. System restores previously saved power settings
2. Sends CURL command to phoropter with restored power
3. Removes "Prev State" from intents list
4. Clears saved state

### When User Clicks Any Other Intent
1. "Prev State" option is removed from intents
2. Normal processing continues

## CURL Commands

### Example: When "Blurry" is clicked (Right Eye)
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"right_eye": {"sph": -0.25}}]}'
```

### Example: When "Prev State" is clicked (Right Eye)
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"right_eye": {"sph": 0.0}}]}'
```

### Example: When "Blurry" is clicked (Left Eye)
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"left_eye": {"sph": -0.25}}]}'
```

### Example: When "Prev State" is clicked (Left Eye)
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"left_eye": {"sph": 0.0}}]}'
```

## Testing

Created comprehensive test script (`test_prev_state.py`) with three test scenarios:

### Test 1: Right Eye Refraction
✓ "Prev State" appears after clicking "Blurry"
✓ Power is reduced by -0.25D when "Blurry" is clicked
✓ Power is restored when "Prev State" is clicked
✓ "Prev State" is removed after use

### Test 2: Left Eye Refraction
✓ "Prev State" appears after clicking "Blurry"
✓ Power is reduced by -0.25D when "Blurry" is clicked
✓ Power is restored when "Prev State" is clicked
✓ "Prev State" is removed after use

### Test 3: Other Intents
✓ "Prev State" does NOT appear for "Unable to read"
✓ "Prev State" does NOT appear for "Able to read"

All tests passed successfully!

## Frontend Integration

No changes needed to the frontend (`app.js`, `index.html`)! The frontend automatically:
1. Receives the updated intents list from the backend
2. Displays "Prev State" as a button when available
3. Sends "Prev State" intent to backend when clicked
4. Updates UI with restored power values

## User Experience Flow

1. **Question**: "I'm covering your left eye. Please read the line you can see clearly."
2. **User clicks**: "Blurry"
3. **System**: Applies -0.25D SPH adjustment
4. **New intents appear**: 
   - Able to read
   - Blurry
   - Unable to read
   - Getting better
   - **Prev State** ← New option!
5. **User clicks**: "Prev State"
6. **System**: Restores previous power settings
7. **Intents updated**: "Prev State" removed, back to standard options

## Notes

- "Prev State" only appears for refraction phases (right_eye_refraction and left_eye_refraction)
- Only one level of undo is supported (saves only the immediately previous state)
- State is cleared when moving to different phases
- The saved state includes all power parameters (SPH, CYL, AXIS) for both eyes, plus occluder and chart state
