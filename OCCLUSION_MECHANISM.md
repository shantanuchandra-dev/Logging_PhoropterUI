# Occlusion Mechanism - Phase A to Phase B Transition

## Summary

**YES, occlusion happens using auxiliary controls (JCC eye mode) during the Phase A to Phase B transition.**

## Detailed Explanation

### Phase A to Phase B Transition Flow

When transitioning from **Phase A (Distance Vision)** to **Phase B (Right Eye Refraction)**, the following sequence occurs:

```python
# In _process_distance_vision() method (line 389-407)

1. Set phase to "right_eye_refraction"
2. Set occluder_state to "Left_Occluded"  
3. Call set_chart() to display snellen_chart_200_150
4. Call set_power(occluder="Left_Occluded")  ← This triggers occlusion
5. Return response to frontend
```

### How Occlusion is Implemented

#### Step 1: `set_power()` is called with `occluder="Left_Occluded"`

```python
# Line 405
self.set_power(occluder="Left_Occluded")
```

#### Step 2: Inside `set_power()`, occluder is mapped to JCC eye mode

```python
# Lines 218-227
if occluder:
    # Note: AuxLens control removed - phoropter handles this automatically
    # Only track occluder state and JCC eye mode mapping
    if occluder == "Left_Occluded":
        jcc_eye_mode = "R"  # Use R when left is occluded (test right eye)
    elif occluder == "Right_Occluded":
        jcc_eye_mode = "L"  # Use L when right is occluded (test left eye)
    elif occluder == "BINO":
        jcc_eye_mode = "BINO"
    self.current_row.occluder_state = occluder
```

#### Step 3: JCC eye mode is set (for non-JCC phases)

```python
# Lines 236-240
# Set JCC eye mode for non-JCC phases only
# JCC phases handle their own state when chart is displayed
if jcc_eye_mode and not is_jcc_phase:
    self.jcc_control(jcc_eye_mode)  ← This sends the occlusion command
    print(f"✓ JCC eye mode set: {jcc_eye_mode}")
```

#### Step 4: `jcc_control()` sends CURL command to phoropter

```python
# Lines 242-251
def jcc_control(self, action: str):
    """Perform JCC action (handle, increase, decrease, etc.)."""
    payload = {"test_cases": [{"jcc": action}]}
    
    cmd = f"""curl -X POST {self.api_endpoint} \
      -H "Content-Type: application/json" \
      -d '{json.dumps(payload)}'"""
    print(f"[CMD] {cmd}")
    subprocess.run(cmd, shell=True, capture_output=True)
    print(f"✓ JCC action: {action}")
```

### Actual CURL Command Sent

When transitioning from Phase A to Phase B (Right Eye Refraction):

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "R"}]}'
```

This command:
- Sets JCC eye mode to **"R"** (Right eye testing)
- **Automatically occludes the left eye** via the phoropter's auxiliary lens control
- Allows testing of the right eye only

## Occluder Mapping

| Occluder State | JCC Eye Mode | Effect |
|----------------|--------------|--------|
| `Left_Occluded` | `"R"` | Left eye occluded, test right eye |
| `Right_Occluded` | `"L"` | Right eye occluded, test left eye |
| `BINO` | `"BINO"` | Both eyes open, binocular testing |

## Why JCC Control Instead of AuxLens?

### Previous Approach (Removed)
Previously, the code explicitly sent `aux_lens` commands:
```json
{
  "test_cases": [{
    "aux_lens": "AuxLensL",  // Occlude right eye
    "right_eye": {...},
    "left_eye": {...}
  }]
}
```

### Current Approach (Using JCC Eye Mode)
Now, the code uses JCC eye mode which **automatically handles occlusion**:
```json
{
  "test_cases": [{
    "jcc": "R"  // Test right eye (automatically occludes left)
  }]
}
```

### Reason for Change
From the code comment (line 219):
```python
# Note: AuxLens control removed - phoropter handles this automatically
# Only track occluder state and JCC eye mode mapping
```

The phoropter's JCC eye mode **automatically manages the auxiliary lens** (occluder), so explicit `aux_lens` commands are not needed.

## Complete Transition Sequence

### Phase A → Phase B (Right Eye Refraction)

```
1. User clicks "Able to read" in Phase A
   ↓
2. Backend: _process_distance_vision() called
   ↓
3. Set occluder_state = "Left_Occluded"
   ↓
4. Call set_chart("snellen_chart_200_150")
   → CURL: {"test_cases": [{"chart": {...}}]}
   ↓
5. Call set_power(occluder="Left_Occluded")
   → Maps to jcc_eye_mode = "R"
   ↓
6. Call jcc_control("R")
   → CURL: {"test_cases": [{"jcc": "R"}]}
   → Phoropter occludes left eye via auxiliary lens
   ↓
7. Frontend displays:
   - Phase: "Phase B: Right Eye Refraction"
   - Question: "I'm covering your left eye..."
   - Occluder: "Left_Occluded"
   - Chart: snellen_chart_200_150
```

### Phase B → Phase D (Left Eye Refraction)

Similarly, when transitioning to left eye refraction:

```
1. Complete duochrome right
   ↓
2. Backend: _transition_to_left_eye_refraction() called
   ↓
3. Set occluder_state = "Right_Occluded"
   ↓
4. Call jcc_control("L")
   → CURL: {"test_cases": [{"jcc": "L"}]}
   → Phoropter occludes right eye via auxiliary lens
   ↓
5. Frontend displays:
   - Phase: "Phase D: Left Eye Refraction"
   - Question: "I'm covering your right eye..."
   - Occluder: "Right_Occluded"
```

## Console Output Example

When transitioning from Phase A to Phase B, you'll see:

```
→ Transitioning to Phase B: Right Eye Refraction (Step 6.1)

[CMD] curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_10"]}}]}'
✓ Displaying: snellen_chart_200_150

[CMD] curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{}]}'
✓ Power set: R(None/None/None) L(None/None/None) Occ: Left_Occluded

[CMD] curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "R"}]}'
✓ JCC action: R
✓ JCC eye mode set: R
```

The third command (`{"jcc": "R"}`) is what triggers the occlusion.

## API Reference

### JCC Eye Mode Command

**Endpoint**: `POST /phoropter/{id}/run-tests`

**Payload**:
```json
{
  "test_cases": [{
    "jcc": "R"  // or "L" or "BINO"
  }]
}
```

**Effect**:
- `"R"`: Test right eye (occludes left eye via auxiliary lens)
- `"L"`: Test left eye (occludes right eye via auxiliary lens)
- `"BINO"`: Binocular mode (both eyes open)

### From curl_API.md (Lines 71-75)

| JCC Eye Mode | Description |
|--------------|-------------|
| **R** | Test Right Eye (Occlude Left) |
| **L** | Test Left Eye (Occlude Right) |
| **BINO** | Binocular mode |

## Verification

To verify occlusion is working:

1. **Start test** and reach Phase A
2. **Click "Able to read"** to transition to Phase B
3. **Check console output** for:
   ```
   [CMD] curl ... '{"test_cases": [{"jcc": "R"}]}'
   ✓ JCC action: R
   ✓ JCC eye mode set: R
   ```
4. **Observe phoropter**: Left eye should be occluded
5. **Frontend should display**: "I'm covering your left eye. Please read..."

## Conclusion

✅ **YES, occlusion happens using auxiliary controls during Phase A to Phase B transition.**

The mechanism is:
1. `set_power(occluder="Left_Occluded")` is called
2. This maps to `jcc_eye_mode = "R"`
3. `jcc_control("R")` sends CURL command: `{"jcc": "R"}`
4. Phoropter automatically occludes left eye via auxiliary lens
5. Right eye testing begins

This approach is cleaner than explicit `aux_lens` commands because the phoropter's JCC eye mode automatically manages the auxiliary lens control.
