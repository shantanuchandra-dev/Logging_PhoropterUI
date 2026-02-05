# JCC Eye Mode Mapping

## Correct Mapping (Updated)

Based on user clarification, the JCC eye mode mapping is:

| Occluder State | Aux Lens | JCC Eye Mode | Clinical Meaning |
|----------------|----------|--------------|------------------|
| `Left_Occluded` | `AuxLensL` | **`L`** | Left eye is occluded, use JCC mode L |
| `Right_Occluded` | `AuxLensR` | **`R`** | Right eye is occluded, use JCC mode R |
| `BINO` | `OFF` | **`BINO`** | Both eyes open |

## Implementation

### For Non-JCC Phases
During Distance Vision, Refraction, Duochrome, and Binocular Balance phases, we set both:
1. **Aux Lens** (to physically occlude the eye)
2. **JCC Eye Mode** (to configure the phoropter)

```python
if occluder == "Left_Occluded":
    payload["test_cases"][0]["aux_lens"] = "AuxLensL"
    jcc_eye_mode = "L"  # Use L when left is occluded
elif occluder == "Right_Occluded":
    payload["test_cases"][0]["aux_lens"] = "AuxLensR"
    jcc_eye_mode = "R"  # Use R when right is occluded
elif occluder == "BINO":
    payload["test_cases"][0]["aux_lens"] = "OFF"
    jcc_eye_mode = "BINO"
```

### For JCC Phases
During JCC Axis and Power phases:
- **Do NOT** set aux lens
- **Do NOT** set JCC eye mode
- The JCC chart handles its own state

```python
is_jcc_phase = self.current_phase in [
    "jcc_axis_right", "jcc_power_right", 
    "jcc_axis_left", "jcc_power_left"
]

# Only set JCC eye mode for non-JCC phases
if jcc_eye_mode and not is_jcc_phase:
    self.jcc_flip(jcc_eye_mode)
```

## Phase-by-Phase Breakdown

### Phase A: Distance Vision
- **Occluder**: BINO
- **JCC Mode**: BINO
- **Purpose**: Assess unaided vision with both eyes

### Phase B: Right Eye Refraction
- **Occluder**: Left_Occluded (left eye is covered)
- **JCC Mode**: L
- **Purpose**: Test right eye while left is occluded

### Phase E: JCC Axis Right
- **Occluder**: Right_Axis_Flip1/Flip2
- **JCC Mode**: (Not set - chart handles it)
- **Purpose**: Refine right eye axis

### Phase F: JCC Power Right
- **Occluder**: Right_Power_Flip1/Flip2
- **JCC Mode**: (Not set - chart handles it)
- **Purpose**: Refine right eye cylinder power

### Phase G: Duochrome Right
- **Occluder**: Left_Occluded
- **JCC Mode**: L
- **Purpose**: Fine-tune right eye sphere

### Phase D: Left Eye Refraction
- **Occluder**: Right_Occluded (right eye is covered)
- **JCC Mode**: R
- **Purpose**: Test left eye while right is occluded

### Phase H: JCC Axis Left
- **Occluder**: Left_Axis_Flip1/Flip2
- **JCC Mode**: (Not set - chart handles it)
- **Purpose**: Refine left eye axis

### Phase I: JCC Power Left
- **Occluder**: Left_Power_Flip1/Flip2
- **JCC Mode**: (Not set - chart handles it)
- **Purpose**: Refine left eye cylinder power

### Phase J: Duochrome Left
- **Occluder**: Right_Occluded
- **JCC Mode**: R
- **Purpose**: Fine-tune left eye sphere

### Phase K: Binocular Balance
- **Occluder**: BINO
- **JCC Mode**: BINO
- **Purpose**: Verify both eyes together

## API Calls

### Example: Distance Vision (BINO)
```bash
# Set power with BINO
curl -X POST .../run-tests \
  -d '{"test_cases": [{"aux_lens": "OFF"}]}'

# Set JCC mode to BINO
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "BINO"}]}'
```

### Example: Right Eye Refraction (Left_Occluded)
```bash
# Set power with Left_Occluded
curl -X POST .../run-tests \
  -d '{"test_cases": [{"right_eye": {"sph": -0.25}, "aux_lens": "AuxLensL"}]}'

# Set JCC mode to L
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "L"}]}'
```

### Example: Left Eye Refraction (Right_Occluded)
```bash
# Set power with Right_Occluded
curl -X POST .../run-tests \
  -d '{"test_cases": [{"left_eye": {"sph": -0.25}, "aux_lens": "AuxLensR"}]}'

# Set JCC mode to R
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "R"}]}'
```

### Example: JCC Phase (No aux_lens or JCC mode calls)
```bash
# Just set the chart - it handles everything
curl -X POST .../run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_19"]}}]}'

# Then use JCC operations (handle, increase, decrease, power_axis_switch)
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "handle"}]}'
```

## Console Output

### Non-JCC Phase (e.g., Right Eye Refraction)
```
✓ Power set: R(-0.25/None/None) L(None/None/None) Occ: Left_Occluded
✓ JCC eye mode set: L
✓ Displaying: snellen_chart_200_150
```

### JCC Phase (e.g., JCC Axis Right)
```
✓ Displaying: jcc_chart
✓ Power set: R(None/None/95) L(None/None/None) Occ: Right_Axis_Flip1
(No JCC eye mode call - chart handles it)
```

## Files Modified

- **`eye_test_engine/interactive_session.py`**
  - Updated `set_power()` to set JCC eye mode for non-JCC phases only
  - Mapping: Left_Occluded → L, Right_Occluded → R, BINO → BINO

## Date
February 5, 2026

## Status
✅ Fixed - JCC eye mode correctly set based on occluder state for non-JCC phases
