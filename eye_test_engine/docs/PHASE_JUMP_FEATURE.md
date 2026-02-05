# Phase Jump Feature

## Overview

The frontend now includes a phase navigation dropdown in the header, allowing you to jump directly to any phase for testing purposes.

---

## UI Location

The phase jump control is located in the **header** of the frontend, visible at all times:

```
┌─────────────────────────────────────────────────────────┐
│  👁️ Eye Test Engine                                     │
│  Interactive Phoropter-Controlled Eye Examination       │
│  ─────────────────────────────────────────────────────  │
│  Jump to Phase: [-- Select Phase --] [Go]              │
└─────────────────────────────────────────────────────────┘
```

---

## Available Phases

| Phase | Name |
|-------|------|
| **Phase A** | Distance Vision (Step 2.1) |
| **Phase B** | Right Eye Refraction (Step 6.1) |
| **Phase E** | JCC Axis Right (Step 6.2) |
| **Phase F** | JCC Power Right (Step 6.2) |
| **Phase G** | Duochrome Right (Step 6.2) |
| **Phase D** | Left Eye Refraction (Step 6.3) |
| **Phase H** | JCC Axis Left (Step 6.4) |
| **Phase I** | JCC Power Left (Step 6.4) |
| **Phase J** | Duochrome Left (Step 6.4) |
| **Phase K** | Binocular Balance (Step 6.5) |

---

## How to Use

### Step 1: Start a Test Session
Click **"Start Eye Test"** button to initialize a session.

### Step 2: Select Target Phase
Use the dropdown in the header to select the phase you want to jump to.

### Step 3: Click "Go"
Click the **"Go"** button to jump to the selected phase.

### Result
- The phoropter is set up for that phase
- The appropriate chart is displayed
- The correct occluder state is set
- Questions and intents for that phase appear
- If it's a JCC phase, auto-flip countdown starts automatically

---

## Use Cases

### Testing Specific Phases
Jump directly to a phase to test its behavior without going through the entire sequence.

**Example:**
- Want to test JCC Axis Right? Jump to Phase E
- Want to test Duochrome? Jump to Phase G or J
- Want to test Binocular Balance? Jump to Phase K

### Debugging
Quickly reproduce issues in specific phases without completing the full test.

### Development
Test new features in specific phases without manual progression.

---

## Implementation

### Frontend (`app.js`)

```javascript
async function jumpToPhase() {
    const select = document.getElementById('phaseSelect');
    const targetPhase = select.value;
    
    // Call backend API
    const response = await fetch(
        `${CONFIG.backendUrl}/api/session/${sessionState.sessionId}/jump`,
        {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phase: targetPhase })
        }
    );
    
    const data = await response.json();
    
    // Update UI
    updateSessionInfo(data);
    displayQuestion(data);
    
    // If auto_flip is requested (JCC phases), start countdown
    if (data.auto_flip) {
        await handleAutoFlip(data.flip_wait_seconds || 2);
    }
}
```

### Backend (`api_server.py`)

```python
@app.route('/api/session/<session_id>/jump', methods=['POST'])
def jump_to_phase(session_id):
    """Jump directly to a specific phase."""
    session = sessions[session_id]
    target_phase = request.json.get('phase')
    
    # Setup the target phase
    session._setup_phase(target_phase)
    state = session._build_response()
    
    return jsonify({
        "session_id": session_id,
        "status": "active",
        **state
    })
```

### Backend (`interactive_session.py`)

The `_setup_phase()` method configures the phoropter for any phase:

```python
def _setup_phase(self, phase: str):
    """Setup phoropter for a specific phase (for testing/jumping)."""
    self.current_phase = phase
    
    if phase == "distance_vision":
        self.set_chart("echart_400")
        self.set_power(occluder="BINO")
        self.current_row.occluder_state = "BINO"
        self.current_row.chart_display = "echart_400"
    
    elif phase == "right_eye_refraction":
        self.set_chart("snellen_chart_20_20_20")
        self.set_power(occluder="Left_Occluded")
        self.current_row.occluder_state = "Left_Occluded"
        self.current_row.chart_display = "snellen_chart_20_20_20"
    
    # ... (all other phases)
```

---

## API Endpoint

### POST `/api/session/<session_id>/jump`

**Request:**
```json
{
  "phase": "jcc_axis_right"
}
```

**Response:**
```json
{
  "session_id": "session_1738167890123",
  "status": "active",
  "phase": "Phase E: JCC Axis Right (Step 6.2)",
  "question": "Focus on the dot chart. This is Flip 1...",
  "intents": [],
  "auto_flip": true,
  "flip_wait_seconds": 2,
  "chart": "jcc_chart",
  "occluder": "Right_Axis_Flip1",
  "power": {
    "right": { "sph": 0.0, "cyl": 0.0, "axis": 180.0 },
    "left": { "sph": 0.0, "cyl": 0.0, "axis": 180.0 }
  }
}
```

---

## Phase Setup Details

### JCC Phases (E, F, H, I)
When jumping to a JCC phase:
1. ✅ Sets `current_phase` to target phase
2. ✅ Displays JCC chart
3. ✅ Sets `jcc_flip_state` to "flip1"
4. ✅ Returns `auto_flip: true` to start countdown
5. ❌ Does NOT call `set_power()` or set aux_lens
6. ❌ Does NOT call JCC eye mode (chart handles it)

### Non-JCC Phases (A, B, D, G, J, K)
When jumping to a non-JCC phase:
1. ✅ Sets `current_phase` to target phase
2. ✅ Displays appropriate chart
3. ✅ Calls `set_power()` with occluder
4. ✅ Sets JCC eye mode (L/R/BINO)
5. ✅ Returns questions and intents

---

## Visual Design

The phase jump control is styled to match the app's design:

- **Dropdown**: Purple border, white background
- **Button**: Purple gradient, white text
- **Hover Effects**: Border color change, button lift
- **Disabled State**: Reduced opacity, no pointer events

---

## Keyboard Shortcuts

The existing keyboard shortcuts still work:
- **1-9**: Select intent by number
- Phase jump requires mouse/touch interaction

---

## Limitations

1. **Session Required**: You must start a test session before jumping to phases
2. **State Reset**: Jumping to a phase resets that phase's state (counters, etc.)
3. **No History**: Jump doesn't preserve previous phase history
4. **Testing Tool**: Intended for development/testing, not production use

---

## Example Usage

### Test JCC Axis Right
1. Start test session
2. Select "Phase E: JCC Axis Right (Step 6.2)" from dropdown
3. Click "Go"
4. **Result**: JCC chart appears, Flip 1 shown, countdown starts

### Test Duochrome Left
1. Start test session
2. Select "Phase J: Duochrome Left (Step 6.4)" from dropdown
3. Click "Go"
4. **Result**: Duochrome chart appears with Red/Green/Both Same intents

### Test Binocular Balance
1. Start test session
2. Select "Phase K: Binocular Balance (Step 6.5)" from dropdown
3. Click "Go"
4. **Result**: Snellen chart with BINO occluder

---

## Files Modified

1. **`frontend/index.html`**
   - Added CSS for `.header-controls` and `.phase-jump`
   - Added phase navigation dropdown in header
   - Added "Go" button

2. **`frontend/app.js`**
   - Added `jumpToPhase()` function
   - Handles API call to `/jump` endpoint
   - Updates UI with new phase state
   - Triggers auto-flip if needed

3. **`api_server.py`**
   - Added `/api/session/<id>/jump` endpoint
   - Calls `session._setup_phase(target_phase)`
   - Returns new phase state

---

## Date
February 5, 2026

## Status
✅ Complete - Phase jump feature implemented in header
