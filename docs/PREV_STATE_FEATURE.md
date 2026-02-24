# Prev State Feature - Complete Documentation

## Overview

The "Prev State" feature provides a one-level undo during eye refraction tests. When a patient clicks "Blurry" or "Unable to read," the system saves the current power state before making the -0.25D adjustment. A "Prev State" button then appears, allowing the patient to revert to the previous settings if the adjustment made things worse.

### Key Capabilities

- **One-Level Undo**: Restores power settings to the state immediately before the last adjustment
- **Phase-Specific**: Only appears during `right_eye_refraction` and `left_eye_refraction`
- **Automatic Cleanup**: Disappears after use or when any other intent is selected
- **Full State Restoration**: Restores SPH, CYL, AXIS for both eyes + occluder + chart

---

## User Experience Flow

### After Clicking "Blurry"

```
┌─────────────────────────────────────────────────────────┐
│ Phase: Phase B: Right Eye Refraction (Step 6.1)        │
├─────────────────────────────────────────────────────────┤
│ Power: Right Eye: -0.25 / 0.00 / 180°  ← Changed!      │
├─────────────────────────────────────────────────────────┤
│   1. Able to read                                       │
│   2. Blurry                                             │
│   3. Unable to read                                     │
│   5. Prev State   ← NEW OPTION!                        │
└─────────────────────────────────────────────────────────┘
```

### After Clicking "Prev State"

```
┌─────────────────────────────────────────────────────────┐
│ Power: Right Eye: 0.00 / 0.00 / 180°   ← Restored!     │
├─────────────────────────────────────────────────────────┤
│   1. Able to read                                       │
│   2. Blurry                                             │
│   3. Unable to read                                     │
│   (Prev State removed)                                  │
└─────────────────────────────────────────────────────────┘
```

### Alternative: After "Blurry" → Click "Able to read"

"Prev State" is silently removed — the adjustment is accepted and testing proceeds normally.

---

## Visual Walkthrough (Both Eyes)

### Right Eye (Left_Occluded)

```
Step 1: Initial           → Right: 0.00/0.00/180
Step 2: Click "Blurry"    → Right: -0.25/0.00/180 + Prev State appears
Step 3: Click "Prev State"→ Right: 0.00/0.00/180  + Prev State removed
```

CURL sent on "Blurry":
```bash
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"right_eye": {"sph": -0.25}}]}'
```

CURL sent on "Prev State":
```bash
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"right_eye": {"sph": 0.0}}]}'
```

### Left Eye (Right_Occluded)

Identical behavior with `left_eye` in the payload.

---

## Technical Implementation

### State Tracking Variables

```python
self.previous_state = None          # Saved state dict
self.show_prev_state_option = False  # Flag to show button
```

### Saved State Structure

```python
{
    'r_sph': 0.0,
    'r_cyl': 0.0,
    'r_axis': 180.0,
    'l_sph': 0.0,
    'l_cyl': 0.0,
    'l_axis': 180.0,
    'occluder_state': 'Left_Occluded',
    'chart_display': 'snellen_chart_20_20_20'
}
```

### Processing Logic (`interactive_session.py`)

#### When "Blurry" is Clicked
1. Save current power state to `self.previous_state`
2. Apply -0.25D SPH adjustment
3. Send CURL to phoropter
4. Set `self.show_prev_state_option = True`
5. Return response — `get_intents()` appends "Prev State"

#### When "Prev State" is Clicked
1. Retrieve `self.previous_state`
2. Restore all power parameters in `self.current_row`
3. Send CURL to phoropter with restored values
4. Clear `self.previous_state = None`
5. Set `self.show_prev_state_option = False`

#### When Any Other Intent is Clicked
1. Clear `self.show_prev_state_option = False`
2. Proceed with normal processing

### `get_intents()` Modification

```python
def get_intents(self) -> List[str]:
    intents = self._get_base_intents()
    if (self.show_prev_state_option
            and self.previous_state is not None
            and self.current_phase in ["right_eye_refraction", "left_eye_refraction"]):
        intents.append("Prev State")
    return intents
```

### `_copy_row_from_dict()` Helper

```python
def _copy_row_from_dict(self, state_dict: dict):
    """Restore RowContext values from a saved state dictionary."""
    self.current_row.r_sph = state_dict['r_sph']
    self.current_row.r_cyl = state_dict['r_cyl']
    self.current_row.r_axis = state_dict['r_axis']
    self.current_row.l_sph = state_dict['l_sph']
    self.current_row.l_cyl = state_dict['l_cyl']
    self.current_row.l_axis = state_dict['l_axis']
    self.current_row.occluder_state = state_dict['occluder_state']
    self.current_row.chart_display = state_dict['chart_display']
```

---

## API Integration

No changes were required to the frontend (`app.js`, `index.html`) or the Flask API server (`api_server.py`). The frontend automatically:
1. Receives the updated intents list containing "Prev State"
2. Displays it as a button
3. Sends "Prev State" intent to backend when clicked
4. Updates the UI with restored power values

---

## Testing

### Test File

`eye_test_engine/tests/test_prev_state.py`

### Test Cases

| Test | Verifies |
|------|---------|
| Right eye refraction — Blurry | "Prev State" appears, power is reduced by -0.25D |
| Right eye refraction — Prev State | Power is restored, "Prev State" is removed |
| Left eye refraction — Blurry | Same behavior for left eye |
| Left eye refraction — Prev State | Same behavior for left eye |
| Other intents | "Prev State" does NOT appear for "Able to read" or "Unable to read" |

### Running Tests

```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests
python test_prev_state.py
```

### Expected Output

```
✓ Right eye refraction: Prev State appears after Blurry
✓ Right eye refraction: Power restored correctly
✓ Left eye refraction: Prev State appears after Blurry
✓ Left eye refraction: Power restored correctly
✓ Prev State does NOT appear for other intents
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Only one level of undo | Prevents confusion from deeper undo chains |
| Only in refraction phases | Other phases have different mechanics |
| Auto-clears on any other intent | Keeps the UI clean; acceptance is implicit |
| No frontend changes needed | Backend-only state management is simpler |

---

**Status:** ✅ Complete — All scenarios tested and verified
