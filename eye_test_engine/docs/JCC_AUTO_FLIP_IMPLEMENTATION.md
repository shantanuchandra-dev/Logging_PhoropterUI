# JCC Automatic Flip Sequence Implementation

## Overview

The JCC (Jackson Cross Cylinder) refinement now features **automatic Flip1 → Flip2 sequence** with a 2-second wait between flips. No patient input is required during the flip presentation - buttons are only enabled after both flips have been shown.

---

## Key Changes

### 1. **Automatic Flip Sequence**

**Old Behavior:**
- Show Flip1, wait for patient response
- Show Flip2, wait for patient response

**New Behavior:**
- Show Flip1 automatically (JCC defaults to Flip1)
- Wait 2 seconds
- Show Flip2 automatically (call `jcc: "handle"`)
- Enable intent buttons for patient to choose

---

### 2. **JCC Setup Sequence**

#### Right Eye Axis
```bash
# 1. Display JCC chart
curl ... -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_19"]}}]}'

# 2. Set to Right eye mode (defaults to Flip1)
curl ... -d '{"test_cases": [{"jcc": "R"}]}'

# 3. Wait 2 seconds (automatic)

# 4. Show Flip2
curl ... -d '{"test_cases": [{"jcc": "handle"}]}'

# 5. Patient responds
```

#### Right Eye Power
```bash
# 1. Switch to Power mode (resets to Flip1)
curl ... -d '{"test_cases": [{"jcc": "power_axis_switch"}]}'

# 2. Wait 2 seconds (automatic)

# 3. Show Flip2
curl ... -d '{"test_cases": [{"jcc": "handle"}]}'

# 4. Patient responds
```

---

### 3. **Patient Response Flow**

After seeing Flip1 and Flip2:

**Flip 1 was better:**
```bash
curl ... -d '{"test_cases": [{"jcc": "increase"}]}'
# Axis: +5°, Power: +0.25D
# Then show Flip1 → Flip2 again
```

**Flip 2 was better:**
```bash
curl ... -d '{"test_cases": [{"jcc": "decrease"}]}'
# Axis: -5°, Power: -0.25D
# Then show Flip1 → Flip2 again
```

**Both Same:**
- Move to next phase (Axis → Power, Power → Duochrome)

**Repeat:**
- Show Flip1 → Flip2 sequence again

---

## Implementation Details

### Backend (interactive_session.py)

#### State Management

```python
self.jcc_flip_state = "showing_flips"  # During auto-flip sequence
```

#### Transition Methods

All JCC transition methods now return `auto_flip_sequence: True`:

```python
def _transition_to_jcc_axis_right(self) -> Dict:
    self.current_phase = "jcc_axis_right"
    self.jcc_flip_state = "showing_flips"
    
    # Setup JCC for right eye axis
    self.set_chart("jcc_chart")
    self.jcc_flip("R")  # Defaults to Flip1
    
    response = self._build_response()
    response["auto_flip_sequence"] = True  # Trigger frontend automation
    return response
```

#### Processing Methods

Simplified to handle only patient responses (no flip state management):

```python
def _process_jcc_axis_right(self, intent: str) -> Dict:
    if "Flip 1" in intent:
        # Increase axis
        self.current_row.r_axis += 5
        self.jcc_flip("increase")
        
        # Show flips again
        response = self._build_response()
        response["auto_flip_sequence"] = True
        return response
    
    elif "Flip 2" in intent:
        # Decrease axis
        self.current_row.r_axis -= 5
        self.jcc_flip("decrease")
        
        # Show flips again
        response = self._build_response()
        response["auto_flip_sequence"] = True
        return response
    
    elif "Both Same" in intent:
        # Move to next phase
        return self._transition_to_jcc_power_right()
    
    elif "Repeat" in intent:
        # Show flips again
        response = self._build_response()
        response["auto_flip_sequence"] = True
        return response
```

---

### Frontend (app.js)

#### Auto-Flip Detection

```javascript
// Check if response includes auto_flip_sequence flag
if (data.auto_flip_sequence) {
    await performJCCFlipSequence(data);
} else {
    enableIntentButtons();
}
```

#### Flip Sequence Function

```javascript
async function performJCCFlipSequence(data) {
    try {
        // Flip1 is already shown (JCC defaults to Flip1)
        showLoading(true, 'Showing Flip 1...');
        addToHistory('JCC Flip 1 shown', 'info');
        await sleep(2000); // Wait 2 seconds
        
        // Show Flip2
        showLoading(true, 'Showing Flip 2...');
        await jccFlip('handle'); // Flip to position 2
        addToHistory('JCC Flip 2 shown', 'info');
        await sleep(500); // Brief pause
        
        // Now enable buttons for patient response
        showLoading(false);
        enableIntentButtons();
        addToHistory('Ready for patient response', 'info');
        
    } catch (error) {
        console.error('Error in JCC flip sequence:', error);
        showError('Failed to perform JCC flip sequence: ' + error.message);
        enableIntentButtons();
    }
}
```

#### JCC Flip API Call

```javascript
async function jccFlip(action) {
    const payload = {
        test_cases: [{
            jcc: action
        }]
    };
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);
    
    const response = await fetch(
        `${CONFIG.phoropterUrl}/phoropter/${CONFIG.phoropterId}/run-tests`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal
        }
    );
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
        throw new Error(`JCC flip failed: ${response.status}`);
    }
}
```

---

### Configuration (protocol.yaml)

Updated questions to reflect automatic sequence:

```yaml
jcc_axis_right:
  questions:
    - "Focus on the dot chart. You will see two options. Which was better: Flip 1 or Flip 2?"
  intents:
    - "Flip 1 was better (GAP Axis - increase axis by 5°)"
    - "Flip 2 was better (RAM Axis - decrease axis by 5°)"
    - "Both Same (no change needed)"
    - "Repeat (show Flip 1 and Flip 2 again)"
```

---

## User Experience

### Timeline

```
0s:  JCC chart displayed
0s:  JCC mode set (R/L)
0s:  Flip1 shown (default)
     Loading: "Showing Flip 1..."
     
2s:  Wait complete
     Loading: "Showing Flip 2..."
     
2.5s: Flip2 shown (handle called)
      Loading: "Ready for patient response"
      
3s:  Buttons enabled
     Patient can now select:
     - "Flip 1 was better"
     - "Flip 2 was better"
     - "Both Same"
     - "Repeat"
```

### History Log

```
12:00:00 - Chart: jcc_chart
12:00:00 - JCC mode: R (Right eye)
12:00:00 - JCC Flip 1 shown
12:00:02 - JCC Flip 2 shown
12:00:03 - Ready for patient response
12:00:10 - Response: Flip 1 was better
12:00:10 - Power: R(-1.25/-0.50/185)
12:00:10 - JCC Flip 1 shown
12:00:12 - JCC Flip 2 shown
12:00:13 - Ready for patient response
```

---

## Complete Flow Example

### Right Eye Axis Refinement

```
1. Transition to JCC Axis Right
   - Display JCC chart
   - Set JCC mode to "R" (Right eye)
   - JCC defaults to Flip1
   
2. Auto-Flip Sequence
   - Show Flip1 (already showing)
   - Wait 2 seconds
   - Call jcc: "handle" (show Flip2)
   - Enable intent buttons
   
3. Patient: "Flip 1 was better"
   - Call jcc: "increase" (axis +5°)
   - Update power on phoropter
   - Trigger auto-flip sequence again
   
4. Auto-Flip Sequence (repeat)
   - Show Flip1
   - Wait 2 seconds
   - Show Flip2
   - Enable buttons
   
5. Patient: "Both Same"
   - Transition to JCC Power Right
   - Call jcc: "power_axis_switch" (resets to Flip1)
   - Trigger auto-flip sequence
   
6. Auto-Flip Sequence (power mode)
   - Show Flip1
   - Wait 2 seconds
   - Show Flip2
   - Enable buttons
   
7. Patient: "Flip 2 was better"
   - Call jcc: "decrease" (power -0.25D)
   - Update power on phoropter
   - Trigger auto-flip sequence again
   
8. Patient: "Both Same"
   - Move to Duochrome
```

---

## Key Points

✅ **No Manual Flip1 Call** - JCC defaults to Flip1 when mode is set  
✅ **Automatic Wait** - 2 seconds between Flip1 and Flip2  
✅ **Single Handle Call** - Only one `jcc: "handle"` call per cycle  
✅ **Mode Reset** - `power_axis_switch` resets to Flip1  
✅ **Buttons Disabled** - During flip sequence  
✅ **Clear Loading States** - "Showing Flip 1...", "Showing Flip 2..."  
✅ **History Logging** - All flip actions logged  
✅ **Error Handling** - Timeouts and error recovery  

---

## Testing

### Test Case 1: Normal Axis Refinement

```
1. Reach JCC Axis Right phase
2. Observe: "Showing Flip 1..." (0s)
3. Observe: "Showing Flip 2..." (2s)
4. Observe: Buttons enabled (3s)
5. Click: "Flip 1 was better"
6. Observe: Axis increased by 5°
7. Observe: Auto-flip sequence repeats
```

### Test Case 2: Power Mode Switch

```
1. In JCC Axis Right, click "Both Same"
2. Observe: Switches to JCC Power Right
3. Observe: power_axis_switch called
4. Observe: Auto-flip sequence starts (Flip1 → Flip2)
5. Buttons enabled after 3 seconds
```

### Test Case 3: Repeat Option

```
1. After seeing Flip1 and Flip2
2. Click: "Repeat"
3. Observe: Auto-flip sequence repeats
4. Flip1 shown → wait 2s → Flip2 shown
5. Buttons enabled
```

---

## Files Modified

1. **interactive_session.py**
   - Updated all JCC transition methods
   - Updated all JCC processing methods
   - Added `auto_flip_sequence` flag to responses

2. **app.js**
   - Added `performJCCFlipSequence()` function
   - Added `jccFlip()` function
   - Updated `submitIntent()` to check for auto-flip
   - Updated `startTest()` to check for auto-flip

3. **protocol.yaml**
   - Updated all JCC phase questions
   - Simplified intents to reflect automatic sequence

4. **JCC_AUTO_FLIP_IMPLEMENTATION.md** (NEW)
   - Complete documentation

---

**JCC automatic flip sequence is now fully implemented!** ✅
