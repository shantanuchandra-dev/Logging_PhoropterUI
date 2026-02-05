# Eye Test Engine - API Usage Guide

This guide shows how to use the Eye Test Engine with the phoropter API for interactive testing.

## Overview

The Eye Test Engine provides:
1. **Conversation Flow** - Questions and intents for each phase
2. **Phoropter Control** - API calls to set power, charts, and occluders
3. **State Management** - Tracks progress through the test

## Quick Start

### 1. View Demo Conversation Flow

```bash
python -m eye_test_engine.demo_conversation
```

This shows all phases with questions and available patient responses.

### 2. Start API Server (Optional)

```bash
python -m eye_test_engine.api_server
```

Server runs on `http://localhost:5000`

## Complete Test Flow with curl Commands

### Phase 1: Distance Vision (Step 2.1)

**Question:** "Please read the line you can see clearly."

**Available Intents:**
1. Able to read
2. Blurry
3. Unable to read

**Setup:**
```bash
# Reset phoropter
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/reset

# Display E-chart 400
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_9"]}}]}'
```

**Machine State:**
- Occluder: BINO
- Chart: echart_400
- Power: R(0/0/180) L(0/0/180)

---

### Phase 2: Right Eye Refraction (RE6.3)

**Question:** "I'm covering your left eye. Please read the line you can see clearly."

**Available Intents:**
1. Able to read
2. Blurry
3. Unable to read
4. Getting better

**Setup:**
```bash
# Display Snellen 20/20/20
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_15"]}}]}'

# Occlude left eye (test right)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "aux_lens": "AuxLensL",
      "right_eye": {"sph": 0.0, "cyl": 0.0, "axis": 180},
      "left_eye": {"sph": 0.0, "cyl": 0.0, "axis": 180}
    }]
  }'
```

**Adjust Power (if "Blurry" or "Unable to read"):**
```bash
# Example: Add -0.50 SPH to right eye
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "aux_lens": "AuxLensL",
      "right_eye": {"sph": -0.50, "cyl": 0.0, "axis": 180},
      "left_eye": {"sph": 0.0, "cyl": 0.0, "axis": 180}
    }]
  }'
```

**Exit Condition:** After 2 consecutive "Unable to read" with SPH changes → Move to JCC Axis

---

### Phase 3: JCC Axis Refinement (Right Eye)

**Questions:**
- Flip 1: "Focus on the dot chart. Is this better? (Flip 1)"
- Flip 2: "Or is this better? (Flip 2)"

**Available Intents (Flip 2 only):**
1. Flip 1: GAP Axis (patient chose Flip 1, increase axis by 5°)
2. Flip 2: RAM Axis (patient chose Flip 2, decrease axis by 5°)
3. Both Same (no change needed)

**Setup:**
```bash
# Display JCC Chart
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_19"]}}]}'

# Set JCC mode to Right eye
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "R"}]}'

# Switch to Axis mode
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "power_axis_switch"}]}'
```

**Flip Sequence:**
```bash
# Flip 1 (show first option)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "handle"}]}'

# Flip 2 (show second option)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "handle"}]}'
```

**If patient chooses Flip 1 (GAP Axis):**
```bash
# Increase axis by 5°
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "increase"}]}'
```

**If patient chooses Flip 2 (RAM Axis):**
```bash
# Decrease axis by 5°
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "decrease"}]}'
```

**Exit Condition:** "Both Same" or no change → Move to JCC Power

---

### Phase 4: JCC Power Refinement (Right Eye)

**Questions:**
- Flip 1: "Focus on the dot chart. Is this better? (Flip 1)"
- Flip 2: "Or is this better? (Flip 2)"

**Available Intents (Flip 2 only):**
1. Flip 1: GAP Power (patient chose Flip 1, increase cylinder by 0.25D)
2. Flip 2: RAM Power (patient chose Flip 2, decrease cylinder by 0.25D)
3. Both Same (no change needed)

**Setup:**
```bash
# Switch to Power mode (if not already)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "power_axis_switch"}]}'
```

**Flip Sequence:** (same as Axis)

**If patient chooses Flip 1 (GAP Power):**
```bash
# Increase cylinder by 0.25D (more positive/less negative)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "increase"}]}'
```

**If patient chooses Flip 2 (RAM Power):**
```bash
# Decrease cylinder by 0.25D (more negative)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "decrease"}]}'
```

**Exception:** If current CYL = 0.00 and patient chooses Flip 1 → "Both Same (no cylinder power)" → Skip to Duochrome

---

### Phase 5: Duochrome (Right Eye)

**Question:** "Which is clearer: red or green, or are they the same?"

**Available Intents:**
1. Red
2. Green
3. Both Same

**Setup:**
```bash
# Display Duochrome chart
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_17"]}}]}'

# Keep left eye occluded (testing right)
```

**Adjustments based on response:**
- **Red clearer:** Add +0.25D SPH (patient slightly myopic)
- **Green clearer:** Add -0.25D SPH (patient slightly hyperopic)
- **Both Same:** Balanced, no adjustment

---

### Phase 6-9: Left Eye (Same as Right Eye)

Repeat Phases 2-5 for left eye:
- **Phase 6:** Left Eye Refraction (occlude right: `"aux_lens": "AuxLensR"`)
- **Phase 7:** JCC Axis Left (set JCC mode to "L")
- **Phase 8:** JCC Power Left
- **Phase 9:** Duochrome Left

---

### Phase 10: Binocular Balance (Step 6.5)

**Question:** "Please read the line you can see clearly."

**Available Intents:**
1. Able to read
2. Blurry
3. Unable to read

**Setup:**
```bash
# Display Snellen 20/20/20
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_15"]}}]}'

# Remove occluder (BINO mode)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "aux_lens": "OFF",
      "right_eye": {"sph": -1.25, "cyl": -0.50, "axis": 90},
      "left_eye": {"sph": -1.00, "cyl": -0.75, "axis": 180}
    }]
  }'
```

**Exit Condition:** Patient can read 20/20 comfortably → Test complete

---

## Using the API Server

### Start Session
```bash
curl -X POST http://localhost:5000/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "patient_001"}'
```

**Response:**
```json
{
  "session_id": "patient_001",
  "status": "started",
  "phase": "Distance Vision (Step 2.1)",
  "question": "Please read the line you can see clearly.",
  "intents": ["Able to read", "Blurry", "Unable to read"],
  "chart": "echart_400",
  "occluder": "BINO",
  "power": {
    "right": {"sph": 0.0, "cyl": 0.0, "axis": 180.0},
    "left": {"sph": 0.0, "cyl": 0.0, "axis": 180.0}
  }
}
```

### Submit Response
```bash
curl -X POST http://localhost:5000/api/session/patient_001/respond \
  -H "Content-Type: application/json" \
  -d '{"intent": "Able to read"}'
```

**Response:**
```json
{
  "session_id": "patient_001",
  "status": "active",
  "phase": "right_eye_refraction",
  "question": "I'm covering your left eye. Please read the line you can see clearly.",
  "intents": ["Able to read", "Blurry", "Unable to read", "Getting better"],
  "chart": "snellen_chart_20_20_20",
  "occluder": "Left_Occluded"
}
```

### Get Session Status
```bash
curl http://localhost:5000/api/session/patient_001/status
```

### End Session
```bash
curl -X POST http://localhost:5000/api/session/patient_001/end
```

**Response:**
```json
{
  "session_id": "patient_001",
  "status": "ended",
  "total_rows": 45,
  "final_prescription": {
    "right_eye": {"sph": -1.25, "cyl": -0.50, "axis": 90, "add": 0.0},
    "left_eye": {"sph": -1.00, "cyl": -0.75, "axis": 180, "add": 0.0}
  }
}
```

---

## Summary

The Eye Test Engine provides a structured conversation flow that:
1. **Asks the right question** for each phase
2. **Provides clear intent options** for the patient
3. **Maps to phoropter API calls** for machine control
4. **Tracks state** through the entire test
5. **Generates final prescription** at the end

Use `demo_conversation.py` to see the full flow, then integrate with your application using the API server or direct phoropter curl commands.
