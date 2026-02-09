# Duochrome Reversal Power Update Fix

## Problem

When a reversal occurred during the Duochrome test (e.g., Green → Red or Red → Green), the CURL command was executed correctly and the internal power state was updated, but the frontend did not display the updated power value.

## Root Cause

The issue was in the `_process_duochrome_right()` and `_process_duochrome_left()` functions in `interactive_session.py`.

When a reversal was detected:
1. The SPH value was correctly updated in `self.current_row`
2. The `jcc_control()` command was sent to the phoropter
3. The function transitioned to the next phase
4. **BUT** the transition function (`_transition_to_left_eye_refraction()` or `_transition_to_binocular_balance()`) intentionally removed the `power` key from the response to prevent unnecessary `setPower` calls

This meant the frontend never received the updated power value and couldn't display it.

## Solution

Modified both `_process_duochrome_right()` and `_process_duochrome_left()` to re-add the `power` key to the response when a reversal occurs:

### Right Eye Duochrome (lines 982-1000)

```python
if reversal:
    # On reversal, transition but include updated power in response
    response = self._transition_to_left_eye_refraction()
    # Re-add power to response so frontend displays updated value
    response['power'] = self._build_response()['power']
    return response
```

### Left Eye Duochrome (lines 1030-1048)

```python
if reversal:
    # On reversal, transition but include updated power in response
    response = self._transition_to_binocular_balance()
    # Ensure power is in response so frontend displays updated value
    if 'power' not in response:
        response['power'] = self._build_response()['power']
    return response
```

## Changes Made

### File: `eye_test_engine/interactive_session.py`

**Right Eye Duochrome (`_process_duochrome_right`)**:
- Lines 982-1000: Added logic to re-add power to response on reversal for both "Red" and "Green" intents

**Left Eye Duochrome (`_process_duochrome_left`)**:
- Lines 1030-1048: Added logic to ensure power is in response on reversal for both "Red" and "Green" intents

## Testing

Created test file: `eye_test_engine/tests/test_duochrome_simple.py`

Test verifies:
1. ✅ Power is included in response after reversal
2. ✅ Power value matches the updated internal state
3. ✅ Frontend will receive and display the correct power value

## Test Results

```
Starting simple duochrome test...
Initial SPH: -1.00D

1. Choosing Green...
   SPH after Green: -0.75D
   Power in response: Yes

2. Choosing Red (reversal)...
   SPH after Red: -1.00D
   Phase: Phase D: Left Eye Refraction (Step 6.3)
   Power in response: Yes
   ✓ SUCCESS: Power included! SPH=-1.00D

✅ Test passed!
```

## Impact

- **Before**: Frontend did not update power display when duochrome reversal occurred
- **After**: Frontend correctly displays updated power value after reversal
- **Backward Compatibility**: No breaking changes; only adds data to response when needed

## Related Files

- `eye_test_engine/interactive_session.py` - Main fix
- `eye_test_engine/tests/test_duochrome_simple.py` - Test file
- `eye_test_engine/tests/test_duochrome_reversal_power.py` - Comprehensive test (optional)

## How to Test Manually

1. Start an eye test session
2. Navigate to Duochrome phase (right or left eye)
3. Choose "Green" (or "Red")
4. Choose "Red" (or "Green") - this triggers reversal
5. **Verify**: Frontend displays the updated SPH value
6. **Verify**: CURL command was sent correctly
7. **Verify**: Phase transitions to next phase

## Technical Notes

- The `_transition_to_left_eye_refraction()` function removes the `power` key to prevent redundant `setPower` calls during normal transitions
- This is correct behavior for "Both Same" intent (no power change)
- But when reversal occurs, power HAS changed, so we need to include it
- The fix preserves the optimization while ensuring correctness
