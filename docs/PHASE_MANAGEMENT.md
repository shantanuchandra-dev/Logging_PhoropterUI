# Phase Management

## Overview

The eye test workflow is organized into 10 phases identified by letters (A–K) and step numbers. Phases flow sequentially but can be jumped to directly for testing and debugging.

---

## Phase Naming Convention

### Phase Map

| Phase ID | Letter | Display Name | Clinical Step |
|----------|--------|-------------|---------------|
| `distance_vision` | **A** | Distance Vision | Step 2.1 |
| `right_eye_refraction` | **B** | Right Eye Refraction | Step 6.1 |
| `jcc_axis_right` | **E** | JCC Axis Right | Step 6.2 |
| `jcc_power_right` | **F** | JCC Power Right | Step 6.2 |
| `duochrome_right` | **G** | Duochrome Right | Step 6.2 |
| `left_eye_refraction` | **D** | Left Eye Refraction | Step 6.3 |
| `jcc_axis_left` | **H** | JCC Axis Left | Step 6.4 |
| `jcc_power_left` | **I** | JCC Power Left | Step 6.4 |
| `duochrome_left` | **J** | Duochrome Left | Step 6.4 |
| `binocular_balance` | **K** | Binocular Balance | Step 6.5 |

### Implementation

```python
self.phase_names = {
    "distance_vision":       "Phase A: Distance Vision (Step 2.1)",
    "right_eye_refraction":  "Phase B: Right Eye Refraction (Step 6.1)",
    "jcc_axis_right":        "Phase E: JCC Axis Right (Step 6.2)",
    "jcc_power_right":       "Phase F: JCC Power Right (Step 6.2)",
    "duochrome_right":       "Phase G: Duochrome Right (Step 6.2)",
    "left_eye_refraction":   "Phase D: Left Eye Refraction (Step 6.3)",
    "jcc_axis_left":         "Phase H: JCC Axis Left (Step 6.4)",
    "jcc_power_left":        "Phase I: JCC Power Left (Step 6.4)",
    "duochrome_left":        "Phase J: Duochrome Left (Step 6.4)",
    "binocular_balance":     "Phase K: Binocular Balance (Step 6.5)",
}
```

Used in `_build_response()`:
```python
phase_display = self.phase_names.get(self.current_phase, self.current_phase)
return {"phase": phase_display, ...}
```

### Phase Flow

```
Phase A → Phase B → Phase E → Phase F → Phase G
                                              ↓
                             Phase K ← Phase J ← Phase I ← Phase H ← Phase D
```

---

## Phase Jump Feature

### Overview

The Phase Jump feature lets users navigate directly to any phase from the frontend UI dropdown. Useful for testing, debugging, and demonstrations.

### Frontend

- Header dropdown listing all 10 phases
- "Go" button triggers the jump
- Located in: `eye_test_engine/frontend/index.html` (header section)
- JavaScript: `jumpToPhase()` in `app.js`

### API Endpoint

**`POST /api/session/<session_id>/jump`**

```json
{ "phase": "jcc_axis_right" }
```

Response: same structure as a normal `respond` call, including `auto_flip` flag for JCC phases.

### What `_setup_phase()` Does

When jumping to a phase:

1. Sets `self.current_phase`
2. Initializes a new `RowContext` (preserves prescription values)
3. Resets phase-specific tracking counters
4. Sets the correct chart and occluder
5. For JCC phases: resets `jcc_flip_state = "flip1"`, calls `set_chart("jcc_chart")`
6. Returns response dict (with `auto_flip: True` for JCC phases)

---

## Phase Setup Details (by Phase)

### distance_vision
- Chart: `echart_400`
- Occluder: `BINO`

### right_eye_refraction
- Chart: `snellen_charts[0]` (largest — `snellen_chart_200_150`)
- Occluder: `Left_Occluded`
- Resets: `current_chart_index = 0`, `unable_read_count = 0`

### left_eye_refraction
- Chart: `snellen_charts[0]`
- Occluder: `Right_Occluded`
- Resets: `current_chart_index = 0`, `unable_read_count = 0`

### jcc_axis_right / jcc_axis_left
- Chart: `jcc_chart`
- Resets: `_reset_jcc_choice_tracking()`
- Returns: `auto_flip: True`, `flip_wait_seconds: 2`

### jcc_power_right / jcc_power_left
- Chart: `jcc_chart`
- Calls: `jcc_control("power_axis_switch")`
- Resets: `_reset_jcc_choice_tracking()`, `jcc_power_zero_flip1_count = 0`
- Returns: `auto_flip: True`

### duochrome_right / duochrome_left
- Chart: `duochrome`
- Resets: `_reset_duochrome_choice_tracking()`

### binocular_balance
- Chart: `snellen_chart_20_20_20`
- Occluder: `BINO`
- Calls: `jcc_control("BINO")`

---

## State Consistency Rule

**Always use `_update_state()` when modifying `occluder_state` or `chart_display`.**

```python
def _update_state(self, occluder: str = None, chart: str = None):
    if occluder is not None:
        self.current_row.occluder_state = occluder
    if chart is not None:
        self.current_row.chart_display = chart
    self.current_row.update_derived_fields()  # Recalculates is_flip1, is_flip2, etc.
```

Skipping `update_derived_fields()` causes derived boolean flags to go stale, which results in missing intents or wrong questions.

---

## API Endpoint Implementation

```python
# api_server.py
@app.route('/api/session/<session_id>/jump', methods=['POST'])
def jump_to_phase(session_id):
    data = request.json
    target_phase = data.get('phase')
    session = sessions[session_id]
    # _setup_phase now returns a response dict with auto_flip flag
    state = session._setup_phase(target_phase)
    return jsonify(state)
```

---

## Checklist for Testing Jump-to-Phase

### Refraction Phases (B & D)
- [ ] Starts with **first chart** (`snellen_chart_200_150`)
- [ ] Correct occluder state (Left_Occluded / Right_Occluded)
- [ ] Chart selector shows correct current chart

### JCC Phases (E, F, H, I)
- [ ] JCC chart displayed
- [ ] Flip 1 shown initially
- [ ] Auto-flips to Flip 2 after 2 seconds
- [ ] Correct eye mode (R for right phases, L for left phases)
- [ ] Choice tracking reset (no false reversals)

### Duochrome Phases (G, J)
- [ ] Duochrome chart displayed
- [ ] Choice tracking reset

### Distance Vision (A)
- [ ] E-chart 400 displayed
- [ ] BINO mode

### Binocular Balance (K)
- [ ] Snellen 20/20/20 displayed
- [ ] BINO mode, JCC set to BINO

---

## Console Output Example

```
→ Transitioning to Phase E: JCC Axis Right (Step 6.2)
[CMD] curl -X POST .../run-tests -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_19"]}}]}'
✓ Displaying: jcc_chart

→ Transitioning to Phase F: JCC Power Right (Step 6.2)
✓ JCC action: power_axis_switch

→ Transitioning to Phase D: Left Eye Refraction (Step 6.3)
✓ Displaying: snellen_chart_200_150
✓ JCC eye mode set: R

→ Transitioning to Phase K: Binocular Balance (Step 6.5)
✓ Displaying: snellen_chart_20_20_20
✓ JCC eye mode set: BINO
```

---

## Files Modified

- `eye_test_engine/interactive_session.py` — `phase_names` dict, `_build_response()`, `_setup_phase()`, all `_transition_to_*()` methods
- `eye_test_engine/api_server.py` — `/api/session/<session_id>/jump` endpoint
- `eye_test_engine/frontend/app.js` — `jumpToPhase()` function
- `eye_test_engine/frontend/index.html` — Phase jump UI (header dropdown)

**Status:** ✅ Complete — All phases correctly initialize on jump
