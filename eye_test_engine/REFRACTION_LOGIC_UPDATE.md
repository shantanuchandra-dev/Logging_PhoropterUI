# Refraction Logic Update - Complete Implementation

## What Was Fixed

The interactive session now implements **proper refraction logic** with chart progression, power adjustments, and state tracking - exactly as used in clinical practice.

---

## Key Changes

### 1. **Chart Progression (Snellen)**

**Correct Order (Big to Small):**
```
snellen_chart_200_150  (biggest - start here)
snellen_chart_100_90
snellen_chart_70_60_50
snellen_chart_40_30_25
snellen_chart_25_20_15
snellen_chart_20_20_20  (target)
snellen_chart_20_15_10  (smallest)
```

**Logic:**
- Start with **largest chart** (200/150)
- Progress to **smaller charts** as patient reads successfully
- Target is **20/20/20**
- Exit to JCC when 20/20/20 is readable OR after 2 "Unable to read" with SPH changes

---

### 2. **Intent-Based Actions (Right/Left Eye Refraction)**

| Intent | Action |
|--------|--------|
| **"Able to read"** | Move to next smaller chart |
| **"Blurry"** | Add -0.25D SPH, stay on same chart |
| **"Unable to read"** | Add -0.25D SPH, stay on same chart, increment counter |
| **"Getting better"** | Keep current power, move to smaller chart |

**Exit Condition:**
- 2 consecutive "Unable to read" responses with SPH changes → Move to JCC Axis
- OR successfully read 20/20/20 → Move to JCC Axis

---

### 3. **JCC Flip Logic (Axis & Power)**

**Flip Sequence:**
1. Show **Flip 1** (no response expected)
2. Wait 2 seconds (simulated)
3. Show **Flip 2** (patient responds)
4. Based on response:
   - **GAP (Flip 1 chosen):** Increase axis/power
   - **RAM (Flip 2 chosen):** Decrease axis/power
   - **Both Same:** Move to next phase
   - **Repeat:** Show Flip 1 and Flip 2 again

**Adjustments:**
- **Axis:** ±5° per cycle
- **Power:** ±0.25D per cycle

**Exit Conditions:**
- "Both Same" selected
- OR "Reverse" option selected (patient changed mind)

---

### 4. **Power Adjustment Values**

| Parameter | Adjustment |
|-----------|------------|
| **Sphere (SPH)** | -0.25D per step |
| **Cylinder (CYL)** | ±0.25D (GAP/RAM) |
| **Axis** | ±5° (GAP/RAM) |

---

### 5. **State Tracking**

New state variables added:
```python
self.snellen_charts = [...]  # Ordered list of charts
self.current_chart_index = 0  # Track position in chart sequence
self.unable_read_count = 0    # Count "Unable to read" responses
self.jcc_flip_state = "flip1" # Track flip1 vs flip2
```

---

## Implementation Details

### Phase Processing Methods

Each phase now has its own processing method:

1. **`_process_distance_vision()`** - Initial baseline
2. **`_process_right_eye_refraction()`** - Right eye sphere refinement with chart progression
3. **`_process_jcc_axis_right()`** - Right eye axis refinement with flip1/flip2
4. **`_process_jcc_power_right()`** - Right eye power refinement with flip1/flip2
5. **`_process_duochrome_right()`** - Right eye red/green balance
6. **`_process_left_eye_refraction()`** - Left eye sphere refinement (same as right)
7. **`_process_jcc_axis_left()`** - Left eye axis refinement
8. **`_process_jcc_power_left()`** - Left eye power refinement
9. **`_process_duochrome_left()`** - Left eye red/green balance
10. **`_process_binocular_balance()`** - Final verification

### Transition Methods

Clean transitions between phases:
- `_transition_to_jcc_axis_right()`
- `_transition_to_jcc_power_right()`
- `_transition_to_duochrome_right()`
- `_transition_to_left_eye_refraction()`
- `_transition_to_jcc_axis_left()`
- `_transition_to_jcc_power_left()`
- `_transition_to_duochrome_left()`
- `_transition_to_binocular_balance()`

### Helper Methods

- `_copy_row_state()` - Copy power values to new row
- `_build_response()` - Build standardized response with current state

---

## Example Flow

### Right Eye Refraction Example

```
1. Start: snellen_chart_200_150, R_SPH=0.0
   Question: "I'm covering your left eye. Please read the line..."
   Patient: "Able to read"
   
2. Next: snellen_chart_100_90, R_SPH=0.0
   Patient: "Blurry"
   
3. Same chart: snellen_chart_100_90, R_SPH=-0.25
   Patient: "Getting better"
   
4. Next: snellen_chart_70_60_50, R_SPH=-0.25
   Patient: "Able to read"
   
5. Next: snellen_chart_40_30_25, R_SPH=-0.25
   Patient: "Unable to read"
   
6. Same chart: snellen_chart_40_30_25, R_SPH=-0.50
   Patient: "Unable to read" (count=2)
   
7. Exit to JCC Axis (2 consecutive "Unable to read")
```

### JCC Axis Example

```
1. Flip1: Right_Axis_Flip1, R_AXIS=180°
   Question: "Focus on the dot chart. Is this better? (Flip 1)"
   Patient: "No response expected"
   
2. Flip2: Right_Axis_Flip2, R_AXIS=180°
   Question: "Or is this better? (Flip 2)"
   Patient: "Flip 1: GAP Axis"
   
3. Adjust: R_AXIS=185° (increased by 5°)
   Back to Flip1: Right_Axis_Flip1, R_AXIS=185°
   
4. Flip2: Right_Axis_Flip2, R_AXIS=185°
   Patient: "Both Same"
   
5. Exit to JCC Power
```

---

## Updated Configuration

### protocol.yaml

Added "Repeat" intent to all JCC phases:

```yaml
jcc_axis_right:
  intents:
    flip2:
      - "Flip 1: GAP Axis (patient chose Flip 1, increase axis by 5°)"
      - "Flip 2: RAM Axis (patient chose Flip 2, decrease axis by 5°)"
      - "Both Same (no change needed)"
      - "Repeat (show Flip 1 and Flip 2 again)"  # NEW

jcc_power_right:
  intents:
    flip2:
      - "Flip 1: GAP Power (patient chose Flip 1, increase cylinder by 0.25D)"
      - "Flip 2: RAM Power (patient chose Flip 2, decrease cylinder by 0.25D)"
      - "Both Same (no change needed)"
      - "Repeat (show Flip 1 and Flip 2 again)"  # NEW
```

---

## Testing

### Test Scenario 1: Normal Progression

```bash
# Start test
./start_frontend.sh

# Distance Vision
Response: "Able to read"

# Right Eye Refraction
Chart: snellen_chart_200_150 → "Able to read"
Chart: snellen_chart_100_90 → "Blurry"
Chart: snellen_chart_100_90 (SPH=-0.25) → "Able to read"
Chart: snellen_chart_70_60_50 → "Able to read"
Chart: snellen_chart_40_30_25 → "Able to read"
Chart: snellen_chart_25_20_15 → "Able to read"
Chart: snellen_chart_20_20_20 → "Able to read"

# JCC Axis Right
Flip1 → "No response expected"
Flip2 → "Flip 1: GAP Axis" (AXIS: 180° → 185°)
Flip1 → "No response expected"
Flip2 → "Both Same"

# JCC Power Right
Flip1 → "No response expected"
Flip2 → "Flip 2: RAM Power" (CYL: 0.0 → -0.25)
Flip1 → "No response expected"
Flip2 → "Both Same"

# Duochrome Right
Response: "Both Same"

# Left Eye Refraction
(Same process as right eye)

# Binocular Balance
Response: "Able to read"

# Test Complete!
```

### Test Scenario 2: Early Exit (Unable to Read)

```bash
# Right Eye Refraction
Chart: snellen_chart_200_150 → "Able to read"
Chart: snellen_chart_100_90 → "Unable to read" (SPH: 0.0 → -0.25)
Chart: snellen_chart_100_90 → "Unable to read" (SPH: -0.25 → -0.50, count=2)

# Exit to JCC Axis (2 consecutive "Unable to read")
```

---

## History Log Example

With the new logic, the history log will show:

```
19:00:00 - Test started
19:00:00 - Phoropter reset to 0/0/180
19:00:01 - Chart: echart_400
19:00:05 - Response: Able to read
19:00:05 - Chart: snellen_chart_200_150
19:00:05 - Power updated - Occluder: Left_Occluded
19:00:10 - Response: Able to read
19:00:10 - Chart: snellen_chart_100_90
19:00:15 - Response: Blurry
19:00:15 - Power: R(-0.25/0.00/180)
19:00:20 - Response: Able to read
19:00:20 - Chart: snellen_chart_70_60_50
... (continues through all phases)
```

---

## Files Modified

1. **`interactive_session.py`** - Complete rewrite of processing logic
   - Added state tracking variables
   - Added 10 phase-specific processing methods
   - Added 8 transition methods
   - Added helper methods

2. **`protocol.yaml`** - Added "Repeat" intent
   - All JCC axis phases
   - All JCC power phases

---

## Benefits

✅ **Clinically Accurate** - Follows actual refraction protocol  
✅ **Chart Progression** - Big to small, proper sequence  
✅ **Power Adjustments** - Correct increments (-0.25D SPH, ±0.25D CYL, ±5° AXIS)  
✅ **Exit Conditions** - Proper triggers for phase transitions  
✅ **State Tracking** - Maintains unable_read_count, chart_index, flip_state  
✅ **Repeat Option** - Allows repeating JCC flips if patient unsure  
✅ **Complete Flow** - All 10 phases properly implemented  

---

## Next Steps

1. **Test the updated flow** - Run `./start_frontend.sh`
2. **Verify chart progression** - Check that charts go from big to small
3. **Verify power adjustments** - Check that SPH/CYL/AXIS change correctly
4. **Verify exit conditions** - Check that phase transitions happen at right times
5. **Test repeat option** - Try "Repeat" during JCC phases

---

**The refraction logic is now complete and clinically accurate!** 🎉
