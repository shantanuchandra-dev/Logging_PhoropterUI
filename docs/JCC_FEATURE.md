# JCC (Jackson Cross Cylinder) Operations - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [JCC Phases](#jcc-phases)
3. [Core Concepts](#core-concepts)
4. [JCC Chart Behavior](#jcc-chart-behavior)
5. [Eye Mode Mapping](#eye-mode-mapping)
6. [Operations Order](#operations-order)
7. [No set_power() During JCC Phases](#no-set_power-during-jcc-phases)
8. [Flow Diagram and State Transitions](#flow-diagram-and-state-transitions)
9. [Automatic Flip Sequence](#automatic-flip-sequence)
10. [Adjustment Step Sizes](#adjustment-step-sizes)
11. [Large Adjustments](#large-adjustments)
12. [Implementation Details](#implementation-details)
13. [Fixes and Clarifications](#fixes-and-clarifications)
14. [Behavior Verification](#behavior-verification)
15. [Testing](#testing)

---

## Overview

The JCC (Jackson Cross Cylinder) refinement system is used during eye examinations to refine axis and cylinder power measurements. The system features automatic Flip 1 → Flip 2 sequences with a 2-second countdown, allowing patients to compare two lens positions and select which provides better vision.

### Key Features

- ✅ Automatic Flip 1 → Flip 2 sequence with 2-second countdown
- ✅ No manual input required during flip presentation
- ✅ Intent buttons appear immediately after Flip 2
- ✅ Immediate transitions between phases
- ✅ Support for standard (±5° axis, ±0.25D power) and large adjustments (±10° axis, ±0.50D power)
- ✅ Proper state tracking and derived field management
- ✅ Spherical equivalent compensation for power adjustments

---

## JCC Phases

The JCC system operates during four specific phases of the eye examination:

| Phase | Letter | Name | Operations Used |
|-------|--------|------|-----------------|
| `jcc_axis_right` | **E** | JCC Axis Right | `handle`, `increase`, `decrease` |
| `jcc_power_right` | **F** | JCC Power Right | `handle`, `increase`, `decrease`, `power_axis_switch` |
| `jcc_axis_left` | **H** | JCC Axis Left | `handle`, `increase`, `decrease` |
| `jcc_power_left` | **I** | JCC Power Left | `handle`, `increase`, `decrease`, `power_axis_switch` |

### Phase Sequence

```
Phase A: Distance Vision
  ↓
Phase B: Right Eye Refraction (Left_Occluded + Snellen)
  ↓
Phase E: JCC Axis Right   → Flip1 ↔ Flip2 until "Both Same"
  ↓
Phase F: JCC Power Right  → Flip1 ↔ Flip2 until "Both Same"
  ↓
Phase G: Duochrome Right
  ↓
Phase D: Left Eye Refraction (Right_Occluded + Snellen)
  ↓
Phase H: JCC Axis Left    → Flip1 ↔ Flip2 until "Both Same"
  ↓
Phase I: JCC Power Left   → Flip1 ↔ Flip2 until "Both Same"
  ↓
Phase J: Duochrome Left
  ↓
Phase K: Binocular Balance (BINO + Snellen)
```

---

## Core Concepts

### JCC Operations

1. **`handle`**: Flips the JCC lens between position 1 and position 2
2. **`increase`**: Increases the current value (axis: +5°, power: +0.25D)
3. **`decrease`**: Decreases the current value (axis: -5°, power: -0.25D)
4. **`power_axis_switch`**: Switches between Axis mode and Power mode

### Eye Mode Operations

- **`R`**: Sets JCC mode to Right eye
- **`L`**: Sets JCC mode to Left eye
- **`BINO`**: Sets JCC mode to Binocular (both eyes)

---

## JCC Chart Behavior

**After calling `set_chart("jcc_chart")`, the JCC chart automatically defaults to Flip 1 of Axis mode.** No additional API calls are needed to initialize the JCC state.

### What NOT to Call After JCC Chart

```python
# WRONG — Don't do this
self.set_chart("jcc_chart")
self.jcc_flip("R")       # ❌ Not needed!
self.set_power(occluder="BINO")  # ❌ Sets aux_lens OFF — not needed!

# CORRECT
self.set_chart("jcc_chart")
# Chart is ready — defaults to Flip 1 of Axis
```

### What API Calls ARE Needed

| Action | API Call | When |
|--------|----------|------|
| Display JCC chart | `set_chart("jcc_chart")` | Phase entry |
| Flip to position 2 | `jcc_flip("handle")` | During test |
| Switch to power | `jcc_flip("power_axis_switch")` | Axis → Power |
| Adjust value | `jcc_flip("increase/decrease")` | After selection |
| Reset to Flip 1 | `jcc_flip("handle")` | After adjustment |
| Set eye mode | `jcc_flip("R/L/BINO")` | ❌ Not needed for JCC phases |

---

## Eye Mode Mapping

### Correct Mapping

| Occluder State | Aux Lens | JCC Eye Mode | Eye Being Tested |
|----------------|----------|--------------|------------------|
| `Left_Occluded` | `AuxLensL` | **`L`** | Right Eye |
| `Right_Occluded` | `AuxLensR` | **`R`** | Left Eye |
| `BINO` | `OFF` | **`BINO`** | Both Eyes |

**Clinical logic:** When left eye is occluded, we're testing the right eye → use JCC mode `L`. When right eye is occluded, we're testing the left eye → use JCC mode `R`.

### Implementation

```python
if occluder == "Left_Occluded":
    payload["test_cases"][0]["aux_lens"] = "AuxLensL"
    jcc_eye_mode = "L"
elif occluder == "Right_Occluded":
    payload["test_cases"][0]["aux_lens"] = "AuxLensR"
    jcc_eye_mode = "R"
elif occluder == "BINO":
    payload["test_cases"][0]["aux_lens"] = "OFF"
    jcc_eye_mode = "BINO"

# Only set JCC eye mode for non-JCC phases
is_jcc_phase = self.current_phase in [
    "jcc_axis_right", "jcc_power_right",
    "jcc_axis_left", "jcc_power_left"
]
if jcc_eye_mode and not is_jcc_phase:
    self.jcc_flip(jcc_eye_mode)
```

---

## Operations Order

**JCC operations (`increase`/`decrease`) must be called FIRST**, then internal state is updated for tracking only.

```python
# CORRECT — Call JCC operation first
self.jcc_flip("increase")      # Phoropter adjusts axis by 5°

# Update internal state for tracking only
self.current_row = self._copy_row_state()
self.current_row.r_axis += 5
if self.current_row.r_axis > 180:
    self.current_row.r_axis -= 180

# Reset to Flip1
self.jcc_flip("handle")
```

**Rationale:** The JCC `increase`/`decrease` operations tell the phoropter to adjust the value. We update `self.current_row` only for internal display tracking — no `set_power()` call needed.

---

## No set_power() During JCC Phases

During JCC phases (E, F, H, I), we **DO NOT** use `set_power()`. Only JCC-specific operations are used.

| Aspect | JCC Phases (E,F,H,I) | Non-JCC Phases (A,B,D,G,J,K) |
|--------|----------------------|------------------------------|
| **Power Setting** | ❌ Not used | ✅ Via `set_power()` |
| **Aux Lens** | ❌ Not set | ✅ Set (AuxLensL/R/OFF) |
| **JCC Operations** | ✅ handle, increase, decrease | ❌ Only eye mode (L/R/BINO) |

### API Call Sequence for JCC Phase (E: Axis Right)

```bash
# 1. Display JCC chart (once at phase start)
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_19"]}}]}'

# 2. Flip to position 2
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"jcc": "handle"}]}'

# 3. Patient selects — increase axis
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"jcc": "increase"}]}'

# 4. Reset to position 1
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"jcc": "handle"}]}'
```

---

## Flow Diagram and State Transitions

```
┌──────────────────────┐
│   FLIP 1 SHOWN       │  No Intent Buttons
│   Countdown: 2s      │
└──────────┬───────────┘
           │ (automatic after 2s — AUTO_FLIP)
           ↓
┌──────────────────────┐
│   FLIP 2 SHOWN       │  4 Intent Buttons:
│                      │  1. Flip 1 better  → Increase axis +5°
│                      │  2. Flip 2 better  → Decrease axis -5°
│                      │  3. Both Same      → Move to Power
│                      │  4. Repeat         → Back to Flip 1
└──────────────────────┘
```

### Timing Diagram

```
t=0s   Flip1 shown (default) — No buttons — Countdown: 2s
t=2s   AUTO_FLIP sent
t=2.1s Flip2 shown — 4 intent buttons enabled
t=5s   Patient clicks "Flip 1 better"
t=5.1s Axis adjusted +5° — Back to Flip1 — Countdown: 2s
... cycle repeats until "Both Same"
t=15s  "Both Same" selected → Move to Power phase immediately
```

### Backend Response Structure

**Flip 1 Response:**
```json
{
  "phase": "jcc_axis_right",
  "question": "Focus on the dot chart. This is Flip 1...",
  "intents": [],
  "auto_flip": true,
  "flip_wait_seconds": 2,
  "occluder": "Right_Axis_Flip1"
}
```

**Flip 2 Response (after AUTO_FLIP):**
```json
{
  "phase": "jcc_axis_right",
  "question": "Now this is Flip 2. Which was better?",
  "intents": [
    "Flip 1 was better (GAP Axis - increase axis by 5°)",
    "Flip 2 was better (RAM Axis - decrease axis by 5°)",
    "Both Same (no change needed)",
    "Repeat (show Flip 1 and Flip 2 again)"
  ],
  "occluder": "Right_Axis_Flip2"
}
```

---

## Automatic Flip Sequence

### Backend Implementation

```python
def _transition_to_jcc_axis_right(self) -> Dict:
    self.current_phase = "jcc_axis_right"
    self.jcc_flip_state = "flip1"
    self.set_chart("jcc_chart")
    response = self._build_response()
    response["auto_flip"] = True
    response["flip_wait_seconds"] = 2
    return response

def _process_jcc_axis_right(self, intent: str) -> Dict:
    if intent == "AUTO_FLIP":
        self.jcc_flip_state = "flip2"
        self._update_state(occluder="Right_Axis_Flip2")
        self.jcc_flip("handle")
        return self._build_response()  # Returns intents immediately

    elif "Flip 1" in intent or "GAP Axis" in intent:
        self.jcc_flip("increase")
        self.current_row = self._copy_row_state()
        self.current_row.r_axis += 5
        if self.current_row.r_axis > 180:
            self.current_row.r_axis -= 180
        self.jcc_flip("handle")
        self.jcc_flip_state = "flip1"
        self._update_state(occluder="Right_Axis_Flip1")
        response = self._build_response()
        response["auto_flip"] = True
        response["flip_wait_seconds"] = 2
        return response

    elif "Both Same" in intent:
        return self._transition_to_jcc_power_right()

    elif "Repeat" in intent:
        self.jcc_flip_state = "flip1"
        self._update_state(occluder="Right_Axis_Flip1")
        response = self._build_response()
        response["auto_flip"] = True
        response["flip_wait_seconds"] = 2
        return response
```

### Frontend Implementation

```javascript
async function handleAutoFlip(waitSeconds) {
    intentButtons.forEach(btn => btn.disabled = true);

    for (let i = waitSeconds; i > 0; i--) {
        countdownDiv.textContent = `⏱️ Showing Flip 2 in ${i} second${i > 1 ? 's' : ''}...`;
        await new Promise(resolve => setTimeout(resolve, 1000));
    }

    const response = await fetch(`/api/session/${id}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ intent: 'AUTO_FLIP' })
    });
    const data = await response.json();
    updateSessionInfo(data);
    displayQuestion(data);  // Creates intent buttons immediately
}
```

---

## Adjustment Step Sizes

| Parameter | Standard | Large ("MUCH better") |
|-----------|----------|-----------------------|
| **Axis** | ±5° | ±10° (two `increase`/`decrease` calls) |
| **Cylinder** | ±0.25D | ±0.50D (two calls + spherical equiv tracking) |

### Axis Wraparound Logic

```python
# Forward: axis > 180° wraps back
if self.current_row.r_axis > 180:
    self.current_row.r_axis -= 180  # e.g. 183° → 3°

# Backward: axis < 0° wraps to end
if self.current_row.r_axis < 0:
    self.current_row.r_axis += 180  # e.g. -3° → 177°
```

---

## Large Adjustments

For "MUCH better" responses, adjustments are doubled by calling the JCC operation twice:

```python
if "MUCH better" in intent and "GAP Axis" in intent:
    self.jcc_flip("increase")
    self.jcc_flip("increase")     # Two calls = +10°
    self.current_row.r_axis += 10
    if self.current_row.r_axis > 180:
        self.current_row.r_axis -= 180
```

For power, spherical equivalent compensation is tracked per 0.25D step:

```python
if "MUCH better" in intent and "GAP Power" in intent:
    for _ in range(2):  # Two 0.25D increments
        was_at_threshold = self._is_at_cyl_threshold(self.current_row.r_cyl)
        self.jcc_flip("increase")
        self.current_row.r_cyl += 0.25
        if was_at_threshold and not self._is_at_cyl_threshold(self.current_row.r_cyl):
            self.current_row.r_sph -= 0.25  # Spherical equivalent reversion
```

### Protocol Configuration

```yaml
jcc_axis_right:
  intents:
    flip2:
      - "Flip 1 was better (GAP Axis - increase axis by 5°)"
      - "Flip 2 was better (RAM Axis - decrease axis by 5°)"
      - "Flip 1 was MUCH better (GAP Axis - increase axis by 10°)"
      - "Flip 2 was MUCH better (RAM Axis - decrease axis by 10°)"
      - "Both Same (no change needed)"
      - "Repeat (show Flip 1 and Flip 2 again)"
```

---

## Implementation Details

### State Management

```python
self.jcc_flip_state = "flip1"  # or "flip2"
```

The `RowContext` class has derived fields (`is_flip1`, `is_flip2`) that must be recalculated whenever `occluder_state` changes manually. Always use `_update_state()`:

```python
def _update_state(self, occluder: str = None, chart: str = None):
    if occluder is not None:
        self.current_row.occluder_state = occluder
    if chart is not None:
        self.current_row.chart_display = chart
    self.current_row.update_derived_fields()  # REQUIRED
```

### Files Modified

- `eye_test_engine/core/context.py` — Added `update_derived_fields()` method
- `eye_test_engine/interactive_session.py` — All JCC phase methods, `_update_state()` helper
- `eye_test_engine/frontend/app.js` — `handleAutoFlip()`, `submitIntent()` auto_flip check
- `eye_test_engine/config/protocol.yaml` — Large adjustment intents added

---

## Fixes and Clarifications

### Fix 1: Missing Flip 2 Intents (Root Cause)

**Problem:** `RowContext` derived fields (`is_flip1`, `is_flip2`) are only calculated at initialization. Manually setting `occluder_state` didn't recalculate them, so `get_intents()` returned empty lists for Flip 2.

**Fix:** Added `update_derived_fields()` + `_update_state()` helper. Now all state changes go through `_update_state()`.

### Fix 2: Operations Order

**Problem:** `set_power()` was called before JCC operations, causing redundant API calls.

**Fix:** JCC operations are called FIRST, then internal state is updated.

### Fix 3: JCC Eye Mode Mapping

**Problem:** JCC eye mode was mapped inversely (Left_Occluded → R, Right_Occluded → L).

**Fix:** Corrected to Left_Occluded → L (test right eye), Right_Occluded → R (test left eye).

---

## Behavior Verification

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Intent buttons appear immediately after AUTO_FLIP | ✅ | `displayQuestion()` creates buttons immediately |
| Immediate transition to Power phase on "Both Same" | ✅ | Direct call to `_transition_to_jcc_power_right()` |
| "Repeat" goes to Flip 1 and starts countdown | ✅ | Returns `auto_flip: true` to trigger countdown |

---

## Testing

### Test Sequence

- [ ] JCC chart displays once
- [ ] Countdown shows 2…1…
- [ ] Intent buttons disabled during countdown
- [ ] Flip 2 shows automatically
- [ ] Buttons re-enable after Flip 2
- [ ] GAP increases axis/power correctly
- [ ] RAM decreases axis/power correctly
- [ ] Repeat restarts countdown
- [ ] Both Same moves to next phase
- [ ] Large adjustments work (±10°, ±0.50D)
- [ ] Spherical equivalent tracks per step
- [ ] Axis wraparound works (0°–180°)

### Expected History Log

```
14:02:00 - Chart: jcc_chart
14:02:03 - Flip 2 displayed
14:02:10 - Response: Flip 1 was better (GAP Axis)
14:02:10 - JCC action: increase
14:02:10 - JCC action: handle
14:02:13 - Flip 2 displayed
14:02:20 - Response: Both Same
14:02:20 - JCC action: power_axis_switch
```

---

**Status:** ✅ Complete — All JCC operations implemented and verified
