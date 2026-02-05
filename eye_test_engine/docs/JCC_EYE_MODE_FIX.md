# JCC Eye Mode Fix

## Issue

When setting occluder states, the JCC eye mode was not being set correctly. The phoropter needs to know which eye is being tested for JCC operations.

## Correct Mapping

### Clinical Logic
- **"I'm covering your LEFT eye"** → Testing RIGHT eye → Use JCC mode `R`
- **"I'm covering your RIGHT eye"** → Testing LEFT eye → Use JCC mode `L`
- **"Both eyes open"** → Testing BINO → Use JCC mode `BINO`

### Technical Mapping
| Occluder State | Aux Lens | JCC Eye Mode | Eye Being Tested |
|----------------|----------|--------------|------------------|
| `Left_Occluded` | `AuxLensL` | `R` | Right Eye |
| `Right_Occluded` | `AuxLensR` | `L` | Left Eye |
| `BINO` | `OFF` | `BINO` | Both Eyes |

## Implementation

### Before
The `set_power()` method only set the `aux_lens` parameter but didn't set the JCC eye mode:

```python
if occluder == "Left_Occluded":
    payload["test_cases"][0]["aux_lens"] = "AuxLensL"
elif occluder == "Right_Occluded":
    payload["test_cases"][0]["aux_lens"] = "AuxLensR"
```

### After
Now `set_power()` automatically sets both the aux lens AND the JCC eye mode:

```python
jcc_eye_mode = None
if occluder:
    if occluder == "Left_Occluded":
        payload["test_cases"][0]["aux_lens"] = "AuxLensL"
        jcc_eye_mode = "R"  # Testing RIGHT eye when left is occluded
    elif occluder == "Right_Occluded":
        payload["test_cases"][0]["aux_lens"] = "AuxLensR"
        jcc_eye_mode = "L"  # Testing LEFT eye when right is occluded
    elif occluder == "BINO":
        payload["test_cases"][0]["aux_lens"] = "OFF"
        jcc_eye_mode = "BINO"
    self.current_row.occluder_state = occluder

# ... (send power command)

# Set JCC eye mode if occluder is specified
if jcc_eye_mode:
    self.jcc_flip(jcc_eye_mode)
    print(f"✓ JCC eye mode set: {jcc_eye_mode}")
```

## API Calls

### Example: Right Eye Refraction (Left Occluded)

**Power Command:**
```bash
curl -X POST https://.../phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"right_eye": {"sph": -0.25}, "aux_lens": "AuxLensL"}]}'
```

**JCC Eye Mode Command (automatically called):**
```bash
curl -X POST https://.../phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "R"}]}'
```

### Example: Left Eye Refraction (Right Occluded)

**Power Command:**
```bash
curl -X POST https://.../phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"left_eye": {"sph": -0.25}, "aux_lens": "AuxLensR"}]}'
```

**JCC Eye Mode Command (automatically called):**
```bash
curl -X POST https://.../phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "L"}]}'
```

## Console Output

### Before Fix
```
✓ Power set: R(-0.25/None/None) L(None/None/None) Occ: Left_Occluded
✓ Displaying: snellen_chart_200_150
```

### After Fix
```
✓ Power set: R(-0.25/None/None) L(None/None/None) Occ: Left_Occluded
✓ JCC eye mode set: R
✓ Displaying: snellen_chart_200_150
```

## Impact

This fix ensures that:
1. ✅ The phoropter knows which eye is being tested
2. ✅ JCC operations (axis/power refinement) work correctly
3. ✅ The correct eye's prescription is adjusted
4. ✅ All occluder state changes automatically set the correct JCC mode

## Files Modified

- **`eye_test_engine/interactive_session.py`**
  - Updated `set_power()` method to automatically call `jcc_flip()` with the correct eye mode

## Testing

Test the following scenarios:

### Test 1: Right Eye Refraction
1. Start eye test
2. Progress to "Right Eye Refraction" phase
3. **Verify console shows:**
   - `Occ: Left_Occluded`
   - `JCC eye mode set: R`

### Test 2: Left Eye Refraction
1. Complete right eye phases
2. Progress to "Left Eye Refraction" phase
3. **Verify console shows:**
   - `Occ: Right_Occluded`
   - `JCC eye mode set: L`

### Test 3: Binocular Balance
1. Complete both eye phases
2. Progress to "Binocular Balance" phase
3. **Verify console shows:**
   - `Occ: BINO`
   - `JCC eye mode set: BINO`

### Test 4: JCC Phases
1. During JCC Axis Right phase
2. **Verify:**
   - Occluder: `Left_Occluded`
   - JCC mode: `R`
   - Flip operations work correctly

## Date
February 5, 2026

## Status
✅ Fixed - JCC eye mode now automatically set based on occluder state
