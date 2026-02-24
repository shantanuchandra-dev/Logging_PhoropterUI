# Eye Test Engine — Complete API Reference

## Overview

This document covers both APIs:
- **Phoropter API** — Direct hardware control (TOPCON phoropter)
- **Eye Test Engine API** — Session management and conversation flow

**Base URLs:**
- Phoropter API: `https://rajasthan-royals.preprod.lenskart.com`
- Eye Test Engine API: `http://localhost:5050`

> Replace `{PHOROPTER_ID}` with your device identifier (configurable in the frontend header).

---

## Quick Start

```bash
# Start API server
python -m eye_test_engine.api_server

# Start a session
curl -X POST http://localhost:5050/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "patient_001"}'
```

---

## Phoropter API Reference

### Reset (To 0/0/180)

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/{PHOROPTER_ID}/reset
```

> **Always reset first** before preloading AR/Lenso values.

---

### Preload AR / Lenso Values

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/{PHOROPTER_ID}/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 1,
      "aux_lens": "BINO",
      "right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90},
      "left_eye": {"sph": -1.75, "cyl": -1.00, "axis": 180}
    }]
  }'
```

---

### Vision Correction (Power Adjustment)

#### Without Previous State

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/{PHOROPTER_ID}/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 1,
      "aux_lens": "AuxLensL",
      "right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90},
      "left_eye": {"sph": -1.75, "cyl": -1.00, "axis": 180}
    }]
  }'
```

| `aux_lens` | Effect |
|-----------|--------|
| `"AuxLensL"` | Occlude Right eye (test Left) |
| `"AuxLensR"` | Occlude Left eye (test Right) |
| `"BINO"` | Both eyes open |
| `"OFF"` | Clear occluder |

#### With Previous State (Recommended)

Providing previous state ensures accurate click calculations:

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/{PHOROPTER_ID}/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 1,
      "prev_aux_lens": "AuxLensL",
      "prev_right_eye": {"sph": 0.00, "cyl": 0.00, "axis": 180},
      "prev_left_eye": {"sph": 0.00, "cyl": 0.00, "axis": 180},
      "aux_lens": "AuxLensL",
      "right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90},
      "left_eye": {"sph": -1.75, "cyl": -1.00, "axis": 180}
    }]
  }'
```

---

### JCC Operations

```bash
# Flip handle (toggle Flip 1 ↔ Flip 2)
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "handle"}]}'

# Switch Axis ↔ Power mode
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "power_axis_switch"}]}'

# Adjust value
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "increase"}]}'
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "decrease"}]}'

# Set eye mode
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "R"}]}'
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "L"}]}'
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "BINO"}]}'
```

---

### Chart Controls

#### Chart 1 Items

| Chart Name | Item ID |
|-----------|---------|
| E-Chart 400 | `chart_9` |
| Snellen 200/150 | `chart_10` |
| Snellen 100/90 | `chart_11` |
| Snellen 70/60/50 | `chart_12` |
| Snellen 40/30/25 | `chart_13` |
| Snellen 20/15/10 | `chart_14` |
| Snellen 20/20/20 | `chart_15` |
| Snellen 25/20/15 | `chart_16` |
| Duochrome | `chart_17` |
| JCC Chart | `chart_19` |
| BINO Chart | `chart_20` |

```bash
# Example: JCC Chart
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_19"]}}]}'

# Example: Snellen with size selection
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_11", "100"]}}]}'
```

#### VA Size Options

| Chart | Sizes |
|-------|-------|
| chart_10 | 200, 150 |
| chart_11 | 100, 80 |
| chart_12 | 70, 60, 50 |
| chart_13 | 40, 30, 25 |
| chart_14 | 20, 15, 10 |
| chart_15 | 20_1, 20_2, 20_3 |
| chart_16 | 25, 20, 15 |
| chart_20 | R, L |

#### Near Vision Chart (Chart 5)

```bash
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"sessionId":"near_vision_test","phoropter_id":"CV-5000PC","test_cases":[{"case_id":1,"chart":{"tab":"Chart5","chart_items":["chart_5"]}}]}'
```

---

### Specialized Controls

```bash
# Pinhole (Alt+V sequence)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/{PHOROPTER_ID}/pinhole

# Occluder (Alt+V sequence)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/{PHOROPTER_ID}/occluder
```

---

### State Synchronization (No Physical Movement)

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/{PHOROPTER_ID}/sync-state \
  -H "Content-Type: application/json" \
  -d '{
    "right_eye": {"sph": -3.00, "cyl": -1.50, "axis": 45},
    "left_eye": {"sph": -2.50, "cyl": -1.25, "axis": 135},
    "aux_lens": "AuxLensR",
    "pd": 64.5
  }'
```

With ADD (near vision):

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/{PHOROPTER_ID}/sync-state \
  -d '{
    "right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90, "add": 1.25},
    "left_eye":  {"sph": -1.75, "cyl": -1.00, "axis": 180, "add": 1.25},
    "aux_lens": "BINO",
    "pd": 64
  }'
```

---

## Eye Test Engine API Reference

### Session Endpoints

#### Start Session

```bash
curl -X POST http://localhost:5050/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "patient_001"}'
```

**Response:**
```json
{
  "session_id": "patient_001",
  "status": "started",
  "phase": "Phase A: Distance Vision (Step 2.1)",
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

#### Submit Response

```bash
curl -X POST http://localhost:5050/api/session/patient_001/respond \
  -H "Content-Type: application/json" \
  -d '{"intent": "Able to read"}'
```

**Response:** Same structure as start, with updated `phase`, `question`, `intents`, `chart`, `occluder`, `power`.

**Auto-flip response (JCC phases):**
```json
{
  "phase": "Phase E: JCC Axis Right (Step 6.2)",
  "intents": [],
  "auto_flip": true,
  "flip_wait_seconds": 2,
  "occluder": "Right_Axis_Flip1"
}
```

#### Get Session Status

```bash
curl http://localhost:5050/api/session/patient_001/status
```

#### Jump to Phase

```bash
curl -X POST http://localhost:5050/api/session/patient_001/jump \
  -H "Content-Type: application/json" \
  -d '{"phase": "jcc_axis_right"}'
```

#### Switch Chart

```bash
curl -X POST http://localhost:5050/api/session/patient_001/switch-chart \
  -H "Content-Type: application/json" \
  -d '{"chart_index": 3}'
```

#### End Session

```bash
curl -X POST http://localhost:5050/api/session/patient_001/end
```

**Response:**
```json
{
  "session_id": "patient_001",
  "status": "ended",
  "final_prescription": {
    "right_eye": {"sph": -1.25, "cyl": -0.50, "axis": 90},
    "left_eye":  {"sph": -1.00, "cyl": -0.75, "axis": 180}
  }
}
```

---

## Complete Test Flow — Phase by Phase

### Phase A: Distance Vision

```bash
# Reset phoropter
curl -X POST .../phoropter/{PHOROPTER_ID}/reset

# Display E-chart 400
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_9"]}}]}'
```

### Phase B: Right Eye Refraction

```bash
# Display Snellen (start with largest)
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_10"]}}]}'

# Occlude left eye (JCC mode R)
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"jcc": "R"}]}'

# If "Blurry" — add -0.25D SPH to right eye
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"aux_lens": "AuxLensL", "right_eye": {"sph": -0.25, "cyl": 0.0, "axis": 180}}]}'
```

### Phases E–F: JCC Axis & Power (Right Eye)

```bash
# Display JCC chart
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_19"]}}]}'

# Auto-flip: Flip 1 → wait 2s → Flip 2 (handle)
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "handle"}]}'

# Patient response: GAP Axis (Flip 1 better)
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "increase"}]}'
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "handle"}]}'  # reset to Flip1

# Axis → Power transition
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "power_axis_switch"}]}'
```

### Phase G: Duochrome Right

```bash
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_17"]}}]}'
```

Adjustments based on response:
- **Red** → Add +0.25D SPH (patient slightly myopic)
- **Green** → Add -0.25D SPH (patient slightly hyperopic)
- **Both Same** → No change

### Phases D, H, I, J: Left Eye (same as B, E, F, G)

Use `"jcc": "L"` and `"aux_lens": "AuxLensR"` for left eye.

### Phase K: Binocular Balance

```bash
# Display BINO chart
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_20"]}}]}'

# Set BINO mode
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests -d '{"test_cases": [{"jcc": "BINO"}]}'
```

---

## Phase Cheat Sheet

| Phase | Occluder | JCC Mode | Chart |
|-------|---------|---------|-------|
| A: Distance Vision | BINO | BINO | echart_400 |
| B: Right Eye Refraction | Left_Occluded | R | snellen (200→20) |
| E: JCC Axis Right | — | — | chart_19 |
| F: JCC Power Right | — | — | chart_19 |
| G: Duochrome Right | Left_Occluded | R | chart_17 |
| D: Left Eye Refraction | Right_Occluded | L | snellen (200→20) |
| H: JCC Axis Left | — | — | chart_19 |
| I: JCC Power Left | — | — | chart_19 |
| J: Duochrome Left | Right_Occluded | L | chart_17 |
| K: Binocular Balance | BINO | BINO | chart_20 |

---

## Previous State Tests (Phoropter API)

### Test 1: Reset and Set from Known State

```bash
curl -X POST .../phoropter/{PHOROPTER_ID}/reset

curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{
    "test_cases": [{
      "prev_aux_lens": "BINO",
      "prev_right_eye": {"sph": 0.00, "cyl": 0.00, "axis": 180},
      "prev_left_eye": {"sph": 0.00, "cyl": 0.00, "axis": 180},
      "aux_lens": "AuxLensL",
      "right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90},
      "left_eye": {"sph": -1.75, "cyl": -1.00, "axis": 180}
    }]
  }'
```

### Test 2: Incremental Change

```bash
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{
    "test_cases": [{
      "prev_aux_lens": "AuxLensL",
      "prev_right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90},
      "prev_left_eye": {"sph": -1.75, "cyl": -1.00, "axis": 180},
      "aux_lens": "AuxLensR",
      "right_eye": {"sph": -3.00, "cyl": -1.50, "axis": 45},
      "left_eye": {"sph": -2.50, "cyl": -1.25, "axis": 135}
    }]
  }'
```

### Test 3: No Change (prev == target)

```bash
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{
    "test_cases": [{
      "prev_aux_lens": "AuxLensR",
      "prev_right_eye": {"sph": -3.00, "cyl": -1.50, "axis": 45},
      "prev_left_eye": {"sph": -2.50, "cyl": -1.25, "axis": 135},
      "aux_lens": "AuxLensR",
      "right_eye": {"sph": -3.00, "cyl": -1.50, "axis": 45},
      "left_eye": {"sph": -2.50, "cyl": -1.25, "axis": 135}
    }]
  }'
```

Expected: Skip all clicks, 0 eye adjustments.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| Phoropter API not responding | Network issue | Check connectivity to preprod URL |
| Session not found | Backend restarted | `POST /api/session/start` again |
| Module not found | Wrong working directory | `cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI` |
| API server won't start | Missing dependencies | `pip install flask flask-cors pyyaml` |

---

**Ready to start?**

```bash
python -m eye_test_engine.api_server
# Then open http://localhost:8080
```
