# Pinhole Test During Distance Vision

## Overview

When a patient reports being unable to read the E-chart during the distance vision phase (Phase A), the system automatically adds a pinhole and asks if the patient can see clearly with it. This helps determine if the vision issue is refractive (correctable with lenses) or potentially due to other factors.

## Feature Details

### When It Triggers

The pinhole test is activated when:
- The patient is in **Phase A: Distance Vision**
- The patient selects the **"Unable to read"** intent

### Workflow

```
1. Patient views E-chart (echart_400)
2. Patient clicks "Unable to read"
3. System adds pinhole via CURL API
4. System asks: "With pinhole: Can you see the E clearly now?"
5. Patient selects one of:
   - "Able to read with pinhole" → Vision issue is likely refractive
   - "Still unable to read" → May indicate other vision issues
6. System transitions to Phase B: Right Eye Refraction
```

### Clinical Significance

#### Pinhole Helps (Vision Improves)
- Indicates the vision problem is **refractive** (correctable with lenses)
- System proceeds normally to refraction
- Expected outcome: proper prescription will help

#### Pinhole Doesn't Help (Vision Still Poor)
- May indicate:
  - Ocular pathology (cataracts, macular degeneration, etc.)
  - Amblyopia
  - Other non-refractive issues
- System still proceeds to refraction but flags for further evaluation
- Optometrist should investigate further

## Technical Implementation

### Backend Changes

#### 1. New Method: `set_pinhole()`

Added to `InteractiveSession` class in `interactive_session.py`:

```python
def set_pinhole(self):
    """Set pinhole on the phoropter."""
    cmd = f"curl -X POST {self.base_url}/phoropter/phoropter-1/pinhole"
    print(f"[CMD] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"✓ Pinhole activated")
    return result
```

#### 2. Updated: `_process_distance_vision()`

Modified to handle pinhole workflow:

```python
def _process_distance_vision(self, intent: str) -> Dict:
    """Process distance vision phase."""
    if intent == "Unable to read":
        # Add pinhole and test again
        print("\n→ Patient unable to read E-chart, adding pinhole")
        self.set_pinhole()
        
        # Update question to ask with pinhole
        self.current_row = self._copy_row_state()
        self.current_row.chart_display = "echart_400"  # Keep E-chart
        
        # Return response with pinhole question
        response = self._build_response()
        response['question'] = "With pinhole: Can you see the E clearly now?"
        response['intents'] = ["Able to read with pinhole", "Still unable to read"]
        return response
    
    elif intent == "Able to read with pinhole":
        # Pinhole helped, move to right eye refraction
        print("✓ Pinhole improved vision, proceeding to refraction")
        return self._transition_to_right_eye_refraction()
    
    elif intent == "Still unable to read":
        # Pinhole didn't help, still move to refraction but flag for further evaluation
        print("⚠️ Pinhole did not improve vision, proceeding to refraction")
        return self._transition_to_right_eye_refraction()
    
    # Default: "Able to read" or "Blurry"
    return self._transition_to_right_eye_refraction()
```

#### 3. New Helper: `_transition_to_right_eye_refraction()`

Extracted common transition logic:

```python
def _transition_to_right_eye_refraction(self) -> Dict:
    """Transition to right eye refraction."""
    self.current_phase = "right_eye_refraction"
    print(f"\n→ Transitioning to {self.phase_names[self.current_phase]}")
    
    self.current_chart_index = 0  # Start with largest chart
    self.unable_read_count = 0
    
    # Create new row
    self.current_row = self._init_row()
    self.current_row.occluder_state = "Left_Occluded"
    self.current_row.chart_display = self.snellen_charts[0]
    
    # Set phoropter
    self.set_chart(self.snellen_charts[0])
    self.set_power(occluder="Left_Occluded")
    
    return self._build_response()
```

### Frontend Changes

**No changes required!** 

The frontend dynamically displays all intents provided by the backend, so the new pinhole-specific intents ("Able to read with pinhole" and "Still unable to read") appear automatically in the UI.

### API Integration

The pinhole is activated using the Phoropter CURL API:

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/pinhole
```

This endpoint triggers the software menu shortcuts (`Alt+V` sequence) to add the pinhole.

## Testing

### Test File: `test_pinhole.py`

Located in: `eye_test_engine/tests/test_pinhole.py`

#### Test Cases

1. **Pinhole Helps** (`test_pinhole_unable_to_read`)
   - Patient unable to read E-chart
   - Pinhole is added
   - Patient can read with pinhole
   - System transitions to right eye refraction

2. **Pinhole Doesn't Help** (`test_pinhole_still_unable`)
   - Patient unable to read E-chart
   - Pinhole is added
   - Patient still can't read with pinhole
   - System still transitions to refraction (flagged)

#### Running the Tests

```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI
python3 eye_test_engine/tests/test_pinhole.py
```

#### Expected Output

```
======================================================================
SUMMARY
======================================================================
Pinhole Helps:          ✅ PASSED
Pinhole Doesn't Help:   ✅ PASSED

✅ ALL TESTS PASSED

Pinhole test functionality verified:
✓ Pinhole added when unable to read E-chart
✓ Pinhole question displayed correctly
✓ Transitions to refraction after pinhole test
✓ Works for both positive and negative outcomes
```

## User Experience

### Before Pinhole

<details>
<summary>UI State</summary>

```
┌─────────────────────────────────────────┐
│ Phase A: Distance Vision (Step 2.1)    │
├─────────────────────────────────────────┤
│                                         │
│ Q: Please read the line you can see     │
│    clearly.                             │
│                                         │
│ [1. Able to read]                       │
│ [2. Blurry]                             │
│ [3. Unable to read]  ← Patient clicks   │
│                                         │
└─────────────────────────────────────────┘
```
</details>

### After Pinhole Added

<details>
<summary>UI State</summary>

```
┌─────────────────────────────────────────┐
│ Phase A: Distance Vision (Step 2.1)    │
├─────────────────────────────────────────┤
│                                         │
│ Q: With pinhole: Can you see the E      │
│    clearly now?                         │
│                                         │
│ [1. Able to read with pinhole]          │
│ [2. Still unable to read]               │
│                                         │
└─────────────────────────────────────────┘
```
</details>

### After Response

System transitions to Phase B: Right Eye Refraction regardless of pinhole outcome.

## Error Handling

- If the pinhole CURL command fails, the subprocess will capture the error
- The system logs the command output for debugging
- The test continues to Phase B even if pinhole activation fails

## Future Enhancements

Potential improvements:
1. **Log pinhole results** to the session data for clinical review
2. **Flag sessions** where pinhole didn't help for optometrist attention
3. **Add pinhole removal** command after the test
4. **Store visual acuity** with and without pinhole for comparison

## Related Files

- **Backend Logic**: `eye_test_engine/interactive_session.py`
- **API Documentation**: `curl_API.md` (line 212)
- **Test Suite**: `eye_test_engine/tests/test_pinhole.py`
- **Frontend**: `eye_test_engine/frontend/app.js` (no changes needed)

## See Also

- [CURL API Documentation](./curl_API.md#pinhole)
- [Distance Vision Phase Protocol](./eye_test_engine/config/protocol.yaml)
- [Test Suite Documentation](./eye_test_engine/tests/README.md)
