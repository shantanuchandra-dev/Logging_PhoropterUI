# BINO Balance - Vision Correction API Update

## Overview

Updated the BINO (Binocular Balance) phase to use the **Vision Correction API with Previous State** for all power adjustments. This ensures accurate click calculations by providing both the previous and new power values to the phoropter.

## Why This Change?

The Vision Correction API with Previous State is more accurate than the simple power setting API because:

1. **Accurate Click Calculation**: The phoropter knows exactly where it's starting from and where it needs to go
2. **State Synchronization**: Prevents drift between the agent's internal state and the phoropter's actual state
3. **Consistency**: Uses the same API pattern as other phases (e.g., transition to left eye refraction)

## Implementation Changes

### Before (Simple Power Setting)

```python
# Old approach - simple power setting
self.current_row.l_sph += 0.25
self.set_power(l_sph=self.current_row.l_sph, occluder="BINO")
```

### After (Vision Correction with Previous State)

```python
# New approach - vision correction with previous state
prev_l_sph = self.current_row.l_sph
new_l_sph = prev_l_sph + 0.25

self.set_power_with_prev_state(
    prev_r_sph=prev_r_sph, prev_r_cyl=prev_r_cyl, prev_r_axis=prev_r_axis,
    prev_l_sph=prev_l_sph, prev_l_cyl=prev_l_cyl, prev_l_axis=prev_l_axis,
    r_sph=prev_r_sph, r_cyl=prev_r_cyl, r_axis=prev_r_axis,
    l_sph=new_l_sph, l_cyl=prev_l_cyl, l_axis=prev_l_axis
)
```

## Updated Intents

All three power-changing intents now use the Vision Correction API:

### 1. Top is blurry [Right Eye]
- **Action**: Add 0.25D to Left Eye SPH
- **API**: Vision Correction with Previous State
- **Example**:
  ```json
  {
    "test_cases": [{
      "case_id": 1,
      "prev_right_eye": { "sph": -1.00, "cyl": -0.50, "axis": 90 },
      "prev_left_eye": { "sph": -1.00, "cyl": -0.50, "axis": 85 },
      "prev_aux_lens": "BINO",
      "right_eye": { "sph": -1.00, "cyl": -0.50, "axis": 90 },
      "left_eye": { "sph": -0.75, "cyl": -0.50, "axis": 85 },
      "aux_lens": "BINO"
    }]
  }
  ```

### 2. Bottom is blurry [Left Eye]
- **Action**: Add 0.25D to Right Eye SPH
- **API**: Vision Correction with Previous State
- **Example**:
  ```json
  {
    "test_cases": [{
      "case_id": 1,
      "prev_right_eye": { "sph": -1.00, "cyl": -0.50, "axis": 90 },
      "prev_left_eye": { "sph": -1.00, "cyl": -0.50, "axis": 85 },
      "prev_aux_lens": "BINO",
      "right_eye": { "sph": -0.75, "cyl": -0.50, "axis": 90 },
      "left_eye": { "sph": -1.00, "cyl": -0.50, "axis": 85 },
      "aux_lens": "BINO"
    }]
  }
  ```

### 3. Prev State
- **Action**: Restore previous power values
- **API**: Vision Correction with Previous State
- **Example**:
  ```json
  {
    "test_cases": [{
      "case_id": 1,
      "prev_right_eye": { "sph": -0.75, "cyl": -0.50, "axis": 90 },
      "prev_left_eye": { "sph": -1.00, "cyl": -0.50, "axis": 85 },
      "prev_aux_lens": "BINO",
      "right_eye": { "sph": -1.00, "cyl": -0.50, "axis": 90 },
      "left_eye": { "sph": -1.00, "cyl": -0.50, "axis": 85 },
      "aux_lens": "BINO"
    }]
  }
  ```

**Note**: The `prev_aux_lens` and `aux_lens` are both set to `"BINO"` because both eyes are open during the binocular balance phase.

## Code Structure

### Top is Blurry Implementation

```python
if intent == "Top is blurry [Right Eye]":
    # Save current state before making changes
    prev_r_sph = self.current_row.r_sph
    prev_r_cyl = self.current_row.r_cyl
    prev_r_axis = self.current_row.r_axis
    prev_l_sph = self.current_row.l_sph
    prev_l_cyl = self.current_row.l_cyl
    prev_l_axis = self.current_row.l_axis
    
    # Save to previous_state for "Prev State" option
    self.previous_state = {
        'r_sph': prev_r_sph,
        'r_cyl': prev_r_cyl,
        'r_axis': prev_r_axis,
        'l_sph': prev_l_sph,
        'l_cyl': prev_l_cyl,
        'l_axis': prev_l_axis,
        'occluder_state': self.current_row.occluder_state,
        'chart_display': self.current_row.chart_display,
    }
    
    # Calculate new state (add 0.25D to left eye)
    new_l_sph = prev_l_sph + 0.25
    
    # Use vision correction API with previous state
    # Both eyes are open in BINO phase, so aux_lens is "BINO"
    self.set_power_with_prev_state(
        prev_r_sph=prev_r_sph, prev_r_cyl=prev_r_cyl, prev_r_axis=prev_r_axis,
        prev_l_sph=prev_l_sph, prev_l_cyl=prev_l_cyl, prev_l_axis=prev_l_axis,
        r_sph=prev_r_sph, r_cyl=prev_r_cyl, r_axis=prev_r_axis,
        l_sph=new_l_sph, l_cyl=prev_l_cyl, l_axis=prev_l_axis,
        prev_aux_lens="BINO",
        aux_lens="BINO"
    )
    
    # Enable "Prev State" option for next response
    self.show_prev_state_option = True
    return self._build_response()
```

### Bottom is Blurry Implementation

Similar structure, but adjusts right eye instead of left eye:

```python
elif intent == "Bottom is blurry [Left Eye]":
    # Save current state
    prev_r_sph = self.current_row.r_sph
    # ... (save all values)
    
    # Calculate new state (add 0.25D to right eye)
    new_r_sph = prev_r_sph + 0.25
    
    # Use vision correction API with previous state
    # Both eyes are open in BINO phase, so aux_lens is "BINO"
    self.set_power_with_prev_state(
        prev_r_sph=prev_r_sph, prev_r_cyl=prev_r_cyl, prev_r_axis=prev_r_axis,
        prev_l_sph=prev_l_sph, prev_l_cyl=prev_l_cyl, prev_l_axis=prev_l_axis,
        r_sph=new_r_sph, r_cyl=prev_r_cyl, r_axis=prev_r_axis,
        l_sph=prev_l_sph, l_cyl=prev_l_cyl, l_axis=prev_l_axis,
        prev_aux_lens="BINO",
        aux_lens="BINO"
    )
```

### Prev State Implementation

Restores the saved previous state:

```python
elif intent == "Prev State":
    if self.previous_state is not None:
        # Get current state before restoring
        curr_r_sph = self.current_row.r_sph
        # ... (get all current values)
        
        # Get previous state values
        prev_r_sph = self.previous_state['r_sph']
        # ... (get all previous values)
        
        # Use vision correction API to restore
        # Both eyes are open in BINO phase, so aux_lens is "BINO"
        self.set_power_with_prev_state(
            prev_r_sph=curr_r_sph, prev_r_cyl=curr_r_cyl, prev_r_axis=curr_r_axis,
            prev_l_sph=curr_l_sph, prev_l_cyl=curr_l_cyl, prev_l_axis=curr_l_axis,
            r_sph=prev_r_sph, r_cyl=prev_r_cyl, r_axis=prev_r_axis,
            l_sph=prev_l_sph, l_cyl=prev_l_cyl, l_axis=prev_l_axis,
            prev_aux_lens="BINO",
            aux_lens="BINO"
        )
        
        self.previous_state = None
        self.show_prev_state_option = False
```

## Benefits

1. **Accurate Click Calculations**: Phoropter knows exact previous and target states
2. **State Synchronization**: Prevents drift between internal state and phoropter state
3. **Consistency**: Uses same API pattern as other phases
4. **Reliability**: More robust than simple power setting

## Files Modified

- `eye_test_engine/interactive_session.py`: Updated `_process_binocular_balance()` method
- `BINO_IMPLEMENTATION.md`: Updated documentation with API examples
- `BINO_VISUAL_GUIDE.md`: Updated API call sequence
- `curl_API.md`: Added BINO chart (chart_20) to chart listing

## Testing

The existing unit tests in `test_binocular_balance_logic.py` continue to pass, as they mock the API calls. The logic remains the same; only the API method used has changed.

## Curl Examples

### Display BINO Chart

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_20"] } }] }'
```

### Adjust Power (Top is Blurry)

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 1,
      "prev_right_eye": { "sph": -1.00, "cyl": -0.50, "axis": 90 },
      "prev_left_eye": { "sph": -1.00, "cyl": -0.50, "axis": 85 },
      "prev_aux_lens": "BINO",
      "right_eye": { "sph": -1.00, "cyl": -0.50, "axis": 90 },
      "left_eye": { "sph": -0.75, "cyl": -0.50, "axis": 85 },
      "aux_lens": "BINO"
    }]
  }'
```

### Adjust Power (Bottom is Blurry)

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 1,
      "prev_right_eye": { "sph": -1.00, "cyl": -0.50, "axis": 90 },
      "prev_left_eye": { "sph": -1.00, "cyl": -0.50, "axis": 85 },
      "prev_aux_lens": "BINO",
      "right_eye": { "sph": -0.75, "cyl": -0.50, "axis": 90 },
      "left_eye": { "sph": -1.00, "cyl": -0.50, "axis": 85 },
      "aux_lens": "BINO"
    }]
  }'
```

### Restore Previous State

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 1,
      "prev_right_eye": { "sph": -0.75, "cyl": -0.50, "axis": 90 },
      "prev_left_eye": { "sph": -1.00, "cyl": -0.50, "axis": 85 },
      "prev_aux_lens": "BINO",
      "right_eye": { "sph": -1.00, "cyl": -0.50, "axis": 90 },
      "left_eye": { "sph": -1.00, "cyl": -0.50, "axis": 85 },
      "aux_lens": "BINO"
    }]
  }'
```

## Summary

✅ **Updated**: All BINO balance power adjustments now use Vision Correction API with Previous State
✅ **Consistent**: Same API pattern as other phases (e.g., left eye refraction transition)
✅ **Accurate**: Provides both previous and new states for precise click calculations
✅ **Tested**: Existing unit tests continue to pass
✅ **Documented**: Updated all documentation with curl examples

This change improves the reliability and accuracy of the BINO balance phase while maintaining consistency with the rest of the codebase.
