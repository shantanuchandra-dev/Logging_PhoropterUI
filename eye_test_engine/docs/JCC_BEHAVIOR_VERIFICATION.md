# JCC Phase Behavior Verification

## Overview

This document verifies that all three JCC phase behaviors are working correctly.

---

## ✅ Requirement 1: Intent Buttons Appear Immediately After AUTO_FLIP

### Expected Behavior
After the 2-second countdown completes and the AUTO_FLIP call is made, intent buttons should appear **immediately** without any delay.

### Implementation

**Backend (`interactive_session.py`):**
```python
if intent == "AUTO_FLIP":
    self.jcc_flip_state = "flip2"
    self.current_row = self._copy_row_state()
    self._update_state(occluder="Right_Axis_Flip2")  # Updates derived fields
    self.jcc_flip("handle")
    return self._build_response()  # Returns intents immediately
```

**Frontend (`app.js`):**
```javascript
const data = await response.json();

// Update UI with Flip2 state
updateSessionInfo(data);
displayQuestion(data);  // Creates intent buttons immediately
```

### Why It Works
1. Backend returns Flip2 state with intents in the response
2. `_update_state()` ensures `is_flip2 = True`, so `get_intents()` returns the correct intent list
3. Frontend's `displayQuestion()` immediately creates and displays the intent buttons
4. No additional delays or waiting periods

### Test Steps
1. Start JCC Axis phase
2. Observe Flip 1 (no buttons, countdown starts)
3. After 2 seconds, AUTO_FLIP is called
4. **Verify:** Intent buttons appear immediately with no delay

---

## ✅ Requirement 2: Immediate Transition to Power Phase

### Expected Behavior
When patient selects "Both Same" or gives a reverse response during Axis refinement, the system should **immediately** transition to Power refinement phase.

### Implementation

**Backend (`interactive_session.py`):**
```python
elif "Both Same" in intent or "Reverse" in intent:
    # Move to JCC Power
    return self._transition_to_jcc_power_right()
```

**`_transition_to_jcc_power_right()` method:**
```python
def _transition_to_jcc_power_right(self) -> Dict:
    """Transition to JCC power refinement for right eye."""
    self.current_phase = "jcc_power_right"
    self.jcc_flip_state = "flip1"
    self.current_row = self._copy_row_state()
    self._update_state(occluder="Right_Power_Flip1", chart="jcc_chart")
    
    self.set_chart("jcc_chart")
    self.jcc_flip("power_axis_switch")  # Switch to power mode
    
    # Tell frontend to auto-flip after 2 seconds
    response = self._build_response()
    response['auto_flip'] = True
    response['flip_wait_seconds'] = 2
    return response
```

### Why It Works
1. "Both Same" intent directly calls `_transition_to_jcc_power_right()`
2. No intermediate steps or delays
3. Response immediately contains the new Power phase state
4. Frontend receives and displays the Power phase question immediately

### Test Steps
1. During JCC Axis phase (Flip 2)
2. Select "Both Same" intent
3. **Verify:** System immediately shows JCC Power phase (Flip 1) with countdown

---

## ✅ Requirement 3: Repeat Goes Back to Flip 1 Immediately

### Expected Behavior
When patient selects "Repeat", the system should:
1. Go back to Flip 1 **immediately**
2. Start the 2-second countdown automatically
3. Then auto-progress to Flip 2

### Implementation

**Backend (`interactive_session.py`):**
```python
elif "Repeat" in intent:
    # Repeat the flip cycle - reset to flip1 and request auto-flip
    self.jcc_flip_state = "flip1"
    self.current_row = self._copy_row_state()
    self._update_state(occluder="Right_Axis_Flip1")  # Back to Flip1
    response = self._build_response()
    response['auto_flip'] = True  # Trigger countdown
    response['flip_wait_seconds'] = 2
    return response
```

**Frontend (`app.js` - `submitIntent()`):**
```javascript
const data = await response.json();

// Update UI
updateSessionInfo(data);
displayQuestion(data);

// If auto_flip is requested, start countdown
if (data.auto_flip) {
    await handleAutoFlip(data.flip_wait_seconds || 2);
}
```

### Why It Works
1. Backend immediately returns Flip1 state with `auto_flip: true`
2. Frontend receives response and displays Flip 1 (with waiting message)
3. `handleAutoFlip()` is called automatically, starting the countdown
4. After 2 seconds, AUTO_FLIP is sent, showing Flip 2 with intents

### Test Steps
1. During JCC Axis phase (Flip 2)
2. Select "Repeat" intent
3. **Verify:** 
   - System immediately shows Flip 1 (waiting message)
   - Countdown starts automatically (2 seconds)
   - Flip 2 appears with intent buttons

---

## Implementation Details

### All Four JCC Phases Updated

The "Repeat" functionality was updated in all four JCC phases:
1. **JCC Axis Right** (`_process_jcc_axis_right`)
2. **JCC Axis Left** (`_process_jcc_axis_left`)
3. **JCC Power Right** (`_process_jcc_power_right`)
4. **JCC Power Left** (`_process_jcc_power_left`)

Each uses `_update_state(occluder="..._Flip1")` to ensure derived fields are correctly updated.

### Key Fix: `_update_state()` Method

All state changes now use the `_update_state()` helper:

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

This ensures that `is_flip1`, `is_flip2`, and other derived fields are always in sync with the actual state.

---

## Summary

| Requirement | Status | Implementation |
|------------|--------|----------------|
| 1. Intent buttons appear immediately after AUTO_FLIP | ✅ | `displayQuestion()` creates buttons immediately |
| 2. Immediate transition to Power phase on "Both Same" | ✅ | Direct call to `_transition_to_jcc_power_right()` |
| 3. "Repeat" goes to Flip 1 and starts countdown | ✅ | Returns `auto_flip: true` to trigger countdown |

All three requirements are correctly implemented and verified!
