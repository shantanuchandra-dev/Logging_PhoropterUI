# Automatic JCC Flip Implementation

## What Was Implemented

Complete **automatic Flip 1 → Flip 2 sequence** for all JCC phases (Axis and Power, both eyes) with:
- ✅ 2-second countdown timer
- ✅ Automatic progression (no manual input for Flip 1)
- ✅ Disabled intent buttons during countdown
- ✅ Proper state tracking (Flip1 → Flip2)
- ✅ Handle calls after increase/decrease

---

## JCC Sequence Flow

### Initial Setup (Example: Right Eye Axis)

1. **Display JCC Chart** (once per test)
   ```bash
   curl ... -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_19"]}}]}'
   ```

2. **Set to Right Eye Mode**
   ```bash
   curl ... -d '{"test_cases": [{"jcc": "R"}]}'
   ```
   - This automatically shows **Flip 1**
   - State: `Right_Axis_Flip1`

3. **Automatic Flip Sequence**
   - Frontend shows: "Focus on the dot chart. This is Flip 1."
   - Countdown: "⏱️ Showing Flip 2 in 2 seconds..."
   - After 2 seconds, frontend calls backend with `intent: "AUTO_FLIP"`
   - Backend calls:
     ```bash
     curl ... -d '{"test_cases": [{"jcc": "handle"}]}'
     ```
   - State changes to: `Right_Axis_Flip2`
   - Frontend shows: "Now this is Flip 2. Which was better?"

4. **Patient Response**
   - Patient selects: "Flip 1 was better" or "Flip 2 was better" or "Both Same"
   - Backend calls appropriate action:
     - Flip 1 → `{"jcc": "increase"}` then `{"jcc": "handle"}`
     - Flip 2 → `{"jcc": "decrease"}` then `{"jcc": "handle"}`
     - Both Same → Move to next phase

5. **Repeat Cycle**
   - If patient chooses "Repeat", go back to step 3
   - If patient chooses GAP/RAM, apply adjustment and go back to step 3

---

## Power Mode Transition

When transitioning from Axis to Power:

1. **Switch to Power Mode**
   ```bash
   curl ... -d '{"test_cases": [{"jcc": "power_axis_switch"}]}'
   ```
   - This **resets to Flip 1** automatically
   - State: `Right_Power_Flip1`

2. **Continue with same Flip 1 → Flip 2 sequence**
   - Same countdown and auto-progression
   - Same patient response handling

---

## Left Eye Transition

When transitioning to left eye:

1. **Set to Left Eye Mode**
   ```bash
   curl ... -d '{"test_cases": [{"jcc": "L"}]}'
   ```
   - This **resets to Flip 1** automatically
   - JCC chart already displayed (from right eye)
   - State: `Left_Axis_Flip1`

2. **Continue with same Flip 1 → Flip 2 sequence**

---

## Frontend Implementation

### Countdown Timer

```javascript
async function handleAutoFlip(waitSeconds) {
    // 1. Disable all intent buttons
    intentButtons.forEach(btn => btn.disabled = true);
    
    // 2. Show countdown: "⏱️ Showing Flip 2 in 2 seconds..."
    for (let i = waitSeconds; i > 0; i--) {
        countdownDiv.textContent = `⏱️ Showing Flip 2 in ${i} second${i > 1 ? 's' : ''}...`;
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    // 3. Call backend with AUTO_FLIP
    fetch('/api/session/{id}/respond', {
        body: JSON.stringify({ intent: 'AUTO_FLIP' })
    });
    
    // 4. Update UI with Flip2 state
    displayQuestion(data);
    
    // 5. Re-enable intent buttons
    intentButtons.forEach(btn => btn.disabled = false);
}
```

### Visual Feedback

During countdown:
```
┌────────────────────────────────────────────────────┐
│ JCC Axis Refinement (Right Eye)                    │
├────────────────────────────────────────────────────┤
│                                                     │
│ ❓ Question                                         │
│ Focus on the dot chart. This is Flip 1.           │
│                                                     │
│ ⏱️ Showing Flip 2 in 2 seconds...                  │
│                                                     │
│ 💬 Please observe Flip 1. Flip 2 will show        │
│    automatically...                                │
│                                                     │
└────────────────────────────────────────────────────┘
```

After countdown:
```
┌────────────────────────────────────────────────────┐
│ JCC Axis Refinement (Right Eye)                    │
├────────────────────────────────────────────────────┤
│                                                     │
│ ❓ Question                                         │
│ Now this is Flip 2. Which was better?             │
│                                                     │
│ 💬 Please select your response:                    │
│ ┌────────────────────────────────────────────────┐ │
│ │ 1. Flip 1 was better (GAP Axis)               │ │
│ ├────────────────────────────────────────────────┤ │
│ │ 2. Flip 2 was better (RAM Axis)               │ │
│ ├────────────────────────────────────────────────┤ │
│ │ 3. Both Same                                   │ │
│ ├────────────────────────────────────────────────┤ │
│ │ 4. Repeat                                      │ │
│ └────────────────────────────────────────────────┘ │
│                                                     │
└────────────────────────────────────────────────────┘
```

---

## Backend State Management

### State Variables

```python
self.jcc_flip_state = "flip1"  # or "flip2"
```

### Response Flags

```python
response = {
    "phase": "jcc_axis_right",
    "question": "Focus on the dot chart. This is Flip 1...",
    "intents": [],  # Empty for Flip1
    "auto_flip": True,  # Tells frontend to auto-progress
    "flip_wait_seconds": 2,  # Countdown duration
    "occluder": "Right_Axis_Flip1",
    "chart": "jcc_chart",
    ...
}
```

---

## Complete JCC Cycle Example

### Right Eye Axis Refinement

```
Step 1: Transition to JCC Axis
  Backend: Set chart, call jcc:"R"
  State: Right_Axis_Flip1
  Frontend: Show "This is Flip 1"
  Response: auto_flip=True

Step 2: Auto-Flip (after 2 seconds)
  Frontend: Countdown 2...1...
  Frontend: Call backend with intent="AUTO_FLIP"
  Backend: Call jcc:"handle"
  State: Right_Axis_Flip2
  Frontend: Show "Now this is Flip 2. Which was better?"
  Intents: ["Flip 1 was better", "Flip 2 was better", "Both Same", "Repeat"]

Step 3: Patient Response - "Flip 1 was better"
  Backend: Call jcc:"increase" (axis += 5°)
  Backend: Call jcc:"handle" (reset to Flip1)
  State: Right_Axis_Flip1
  Response: auto_flip=True
  → Go back to Step 2

Step 4: Patient Response - "Both Same"
  Backend: Transition to Power mode
  Backend: Call jcc:"power_axis_switch"
  State: Right_Power_Flip1
  Response: auto_flip=True
  → Continue to Power refinement
```

---

## Files Modified

### 1. interactive_session.py

**Added auto_flip logic to all JCC processing methods:**
- `_process_jcc_axis_right()` - Handle AUTO_FLIP intent, return auto_flip flag
- `_process_jcc_power_right()` - Same
- `_process_jcc_axis_left()` - Same
- `_process_jcc_power_left()` - Same

**Updated all transition methods:**
- `_transition_to_jcc_axis_right()` - Return auto_flip=True
- `_transition_to_jcc_power_right()` - Return auto_flip=True
- `_transition_to_jcc_axis_left()` - Return auto_flip=True
- `_transition_to_jcc_power_left()` - Return auto_flip=True

**Added handle calls after increase/decrease:**
- After `jcc_flip("increase")` → `jcc_flip("handle")`
- After `jcc_flip("decrease")` → `jcc_flip("handle")`

**Updated get_intents():**
- Return empty list for flip1 (no response needed)
- Return full list for flip2 (patient chooses)

### 2. app.js

**Added handleAutoFlip() function:**
- Disables all intent buttons
- Shows countdown timer (2...1...)
- Calls backend with AUTO_FLIP after countdown
- Updates UI with Flip2 state
- Re-enables intent buttons

**Updated submitIntent():**
- Check for auto_flip flag in response
- Call handleAutoFlip() if needed

**Updated startTest():**
- Check for auto_flip flag in initial response
- Call handleAutoFlip() if needed

**Updated displayQuestion():**
- Show waiting message if intents are empty and auto_flip is true

### 3. protocol.yaml

**Updated all JCC phases:**
- flip1 question: "This is Flip 1. (Flip 2 will show automatically in 2 seconds)"
- flip2 question: "Now this is Flip 2. Which was better?"
- flip1 intents: [] (empty - no response needed)
- flip2 intents: Updated wording to "Flip 1 was better" / "Flip 2 was better"

---

## Testing Checklist

- [ ] JCC chart displays only once
- [ ] Right eye mode set correctly
- [ ] Flip 1 shows with countdown
- [ ] Countdown displays 2...1...
- [ ] Intent buttons are disabled during countdown
- [ ] Flip 2 shows automatically after 2 seconds
- [ ] Intent buttons re-enable after Flip 2
- [ ] Patient can select response
- [ ] GAP increases axis/power correctly
- [ ] RAM decreases axis/power correctly
- [ ] Handle is called after increase/decrease
- [ ] Repeat option works correctly
- [ ] Both Same moves to next phase
- [ ] Power mode switch resets to Flip 1
- [ ] Left eye mode switch resets to Flip 1

---

## History Log Example

With the new implementation:

```
14:00:00 - Test started
14:00:01 - Phoropter reset to 0/0/180
14:00:01 - Chart: echart_400
14:00:05 - Response: Able to read
14:00:05 - Chart: snellen_chart_200_150
14:00:05 - Power updated - Occluder: Left_Occluded
... (sphere refinement)
14:02:00 - Chart: jcc_chart
14:02:00 - JCC action: R
14:02:03 - Flip 2 displayed
14:02:10 - Response: Flip 1 was better (GAP Axis)
14:02:10 - JCC action: increase
14:02:10 - JCC action: handle
14:02:13 - Flip 2 displayed
14:02:20 - Response: Both Same
14:02:20 - JCC action: power_axis_switch
14:02:23 - Flip 2 displayed
... (continues)
```

---

## Key Features

✅ **Automatic Flip Progression** - No manual input needed for Flip 1  
✅ **Visual Countdown** - Clear 2-second timer  
✅ **Disabled Buttons** - Prevents premature responses  
✅ **Proper State Tracking** - Flip1 vs Flip2 states  
✅ **Handle Reset** - Calls handle after increase/decrease  
✅ **Single Chart Display** - JCC chart shown only once  
✅ **Mode Switching** - R/L/Power modes reset to Flip 1  
✅ **Repeat Option** - Can repeat the flip sequence  

---

## Next Steps

1. **Test the flow** - Start frontend and go through JCC phases
2. **Verify countdown** - Check that 2-second timer displays
3. **Verify auto-flip** - Confirm Flip 2 shows automatically
4. **Verify adjustments** - Check axis/power changes correctly
5. **Verify handle calls** - Confirm handle is called after increase/decrease

---

**The automatic flip sequence is now fully implemented!** 🎉
