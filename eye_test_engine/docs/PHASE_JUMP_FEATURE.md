# Phase Jump Feature

## Overview
The Phase Jump feature allows users to directly navigate to any phase of the eye test from the frontend UI, bypassing the normal sequential flow. This is useful for testing, debugging, and demonstrations.

## How It Works

### Frontend (UI)
1. **Phase Selector**: A dropdown menu in the header displays all available phases
2. **Jump Button**: A "Go" button triggers the phase jump
3. **Location**: `eye_test_engine/frontend/index.html` (header section)

### Backend (API)
1. **Endpoint**: `POST /api/session/<session_id>/jump`
2. **Payload**: `{ "phase": "phase_name" }`
3. **Response**: Returns the same structure as a normal response, including:
   - `phase`: Current phase name (e.g., "Phase E: JCC Axis Right")
   - `question`: The question to ask the patient
   - `intents`: Available response options
   - `chart`: Current chart displayed
   - `occluder`: Current occluder state
   - `power`: Current prescription values for both eyes

## Available Phases

| Phase ID | Display Name |
|----------|-------------|
| `distance_vision` | Phase A: Distance Vision |
| `right_eye_refraction` | Phase B: Right Eye Refraction |
| `jcc_axis_right` | Phase E: JCC Axis Right |
| `jcc_power_right` | Phase F: JCC Power Right |
| `duochrome_right` | Phase G: Duochrome Right |
| `left_eye_refraction` | Phase D: Left Eye Refraction |
| `jcc_axis_left` | Phase H: JCC Axis Left |
| `jcc_power_left` | Phase I: JCC Power Left |
| `duochrome_left` | Phase J: Duochrome Left |
| `binocular_balance` | Phase K: Binocular Balance |

## Implementation Details

### Phase Setup (`_setup_phase` method)
When jumping to a phase, the backend:

1. **Sets Current Phase**: Updates `self.current_phase` to the target phase
2. **Creates New Row**: Initializes a new `RowContext` for the phase
3. **Copies Power Values**: Preserves prescription values from the previous phase
4. **Resets Refraction State**: Clears chart index and unable-read count
5. **Phase-Specific Initialization**:
   - Sets appropriate chart (e.g., `snellen_chart_20_20_20`, `jcc_chart`, `duochrome`)
   - Sets occluder state (e.g., `BINO`, `Left_Occluded`, `Right_Occluded`)
   - For JCC phases:
     - Sets `self.jcc_flip_state = "flip1"`
     - For power phases, calls `self.jcc_flip("power_axis_switch")`
   - Updates state using `self._update_state()` to ensure derived fields are consistent

### State Consistency
The `_update_state()` helper method ensures that:
- `occluder_state` and `chart_display` are updated in `RowContext`
- Derived fields (`is_flip1`, `is_flip2`, `is_jcc_axis`, `is_jcc_power`) are recalculated
- This prevents issues where intents don't appear due to stale derived fields

### Response Generation
After phase setup, `_build_response()` is called to:
1. Get the appropriate question using `get_question()`
2. Get available intents using `get_intents()`
3. Return all state information to the frontend

## Usage

### For Testing
1. Start a test session
2. Select a phase from the dropdown in the header
3. Click "Go"
4. The UI will immediately show the question and intents for that phase

### For Debugging
- Jump to specific phases to test their logic
- Verify that questions and intents are correct for each phase
- Test JCC flip sequences without going through the entire test

## Code Locations

### Backend
- **Phase Setup**: `eye_test_engine/interactive_session.py` → `_setup_phase()` method (lines ~925-1000)
- **API Endpoint**: `eye_test_engine/api_server.py` → `/api/session/<session_id>/jump` (lines ~87-110)
- **State Update**: `eye_test_engine/interactive_session.py` → `_update_state()` method (lines ~917-923)

### Frontend
- **UI Elements**: `eye_test_engine/frontend/index.html` → Header section
- **Jump Function**: `eye_test_engine/frontend/app.js` → `jumpToPhase()` (lines ~453-502)
- **Display Logic**: `eye_test_engine/frontend/app.js` → `displayQuestion()` (lines ~181-211)

## Important Notes

### JCC Phases
- When jumping to JCC axis phases, the chart defaults to Flip 1 of Axis
- No explicit `jcc_flip("R")` or `jcc_flip("L")` calls are needed after `set_chart("jcc_chart")`
- `AuxLens OFF` is not called after JCC chart display
- For JCC power phases, `jcc_flip("power_axis_switch")` is called to switch from axis to power mode

### State Preservation
- Prescription values (SPH, CYL, AXIS) are preserved when jumping between phases
- This allows testing later phases with realistic prescription values

### Derived Fields
- Always use `_update_state()` when modifying `occluder_state` or `chart_display`
- This ensures `RowContext`'s derived fields remain consistent
- Failure to do so can result in missing intents or incorrect questions
