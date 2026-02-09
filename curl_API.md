# 📡 Phoropter API Reference (Preprod)

This document provides complete `curl` commands for controlling the TOPCON phoropter remotely via the preprod broker.

**Base URL:** `https://rajasthan-royals.preprod.lenskart.com`
**Phoropter ID:** `phoropter-1`

---

## 1. Vision Correction (Power Adjustments)

### Set Eyes & Occluder (Combined)
Use the `run-tests` endpoint to set power for both eyes and (optionally) an occluder in a single request.

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "case_id": 1,
        "aux_lens": "AuxLensL",
        "right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90},
        "left_eye": {"sph": -1.75, "cyl": -1.00, "axis": 180}
      }
    ]
  }'
```

| Parameter | Description |
| :--- | :--- |
| **aux_lens** | (Optional) "AuxLensR" (occlude L), "AuxLensL" (occlude R), or "OFF" |
| **right_eye** / **left_eye** | Objects containing `sph`, `cyl`, and `axis` |

#### With Previous State (Recommended)
To ensure accurate click calculations, provide the **previous state** along with the target state. This is especially important when the agent's internal state might be out of sync.

```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [
      {
        "case_id": 1,
        "prev_aux_lens": "AuxLensL",
        "prev_right_eye": {"sph": 0.00, "cyl": 0.00, "axis": 180},
        "prev_left_eye": {"sph": 0.00, "cyl": 0.00, "axis": 180},
        "aux_lens": "AuxLensL",
        "right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90},
        "left_eye": {"sph": -1.75, "cyl": -1.00, "axis": 180}
      }
    ]
  }'
```

| Parameter | Description |
| :--- | :--- |
| **aux_lens** | (Optional) "AuxLensL" (JCC R mode), "AuxLensR" (JCC L mode), or "BINO" (binocular) |
| **right_eye** / **left_eye** | Objects containing `sph`, `cyl`, and `axis` |
| **prev_aux_lens** | (Optional) Previous occluder state - used as starting point for calculations |
| **prev_right_eye** / **prev_left_eye** | (Optional) Previous eye values - used as starting point for click calculations |

> **Note:** `aux_lens` values are mapped to JCC commands:
> - `"AuxLensL"` → JCC R mode (tests Right eye, occludes Left)
> - `"AuxLensR"` → JCC L mode (tests Left eye, occludes Right)
> - `"BINO"` → JCC BINO mode (binocular testing)

---

## 2. JCC & Auxiliary Controls
These commands utilize the `run-tests` endpoint to trigger specific UI interactions.

### JCC Operations (Handle, Toggle & Adjust)
| Action | Description |
| :--- | :--- |
| **JCC Handle** | Flip the JCC lens handle |
| **Power/Axis Switch** | Toggle between Power and Axis mode |
| **Increase** | Increase value in JCC mode |
| **Decrease** | Decrease value in JCC mode |

```bash
# JCC Handle
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "jcc": "handle" }] }'

# Power/Axis Switch
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "jcc": "power_axis_switch" }] }'

# Increase
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "jcc": "increase" }] }'

# Decrease
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "jcc": "decrease" }] }'
```

### JCC Eye Modes (L, R, BINO)
| Mode | Action |
| :--- | :--- |
| **R** | Test Right Eye (Occlude Left) |
| **L** | Test Left Eye (Occlude Right) |
| **BINO** | Binocular mode |

```bash
# Set mode to RIGHT
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "jcc": "R" }] }'

# Set mode to LEFT
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "jcc": "L" }] }'

# Set mode to BINO
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "jcc": "BINO" }] }'
```

---

## 3. Chart Controls
Individual commands for specific chart items on **Chart1** and **Chart2**.

### Chart 1: Visual Acuity & Tests
| Action | Item ID | Description |
| :--- | :--- | :--- |
| **echart_400** | chart_9 | E-Chart 400 |
| **snellen_chart_200_150** | chart_10 | Snellen 200/150 |
| **snellen_chart_100_90** | chart_11 | Snellen 100/90 |
| **snellen_chart_70_60_50** | chart_12 | Snellen 70/60/50 |
| **snellen_chart_40_30_25** | chart_13 | Snellen 40/30/25 |
| **snellen_chart_20_15_10** | chart_14 | Snellen 20/15/10 |
| **snellen_chart_20_20_20** | chart_15 | Snellen 20/20/20 |
| **snellen_chart_25_20_15** | chart_16 | Snellen 25/20/15 |
| **Duochrome** | chart_17 | Duochrome Test |
| **JCC Chart** | chart_19 | JCC Cross Cylinder Chart |

#### Individual Commands (Chart 1)

```bash
# echart_400
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_9"] } }] }'

# snellen_chart_200_150
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_10"] } }] }'

# snellen_chart_100_90
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_11"] } }] }'

# snellen_chart_70_60_50
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_12"] } }] }'

# snellen_chart_40_30_25
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_13"] } }] }'

# snellen_chart_20_15_10
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_14"] } }] }'

# snellen_chart_20_20_20
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_15"] } }] }'

# snellen_chart_25_20_15
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_16"] } }] }'

# Duochrome
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_17"] } }] }'

# JCC Chart
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart1", "chart_items": ["chart_19"] } }] }'
```

### Chart 2: Miscellaneous
```bash
# Show chart_1, chart_2, and chart_3 from Chart2 tab
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{ "test_cases": [{ "chart": { "tab": "Chart2", "chart_items": ["chart_1", "chart_2", "chart_3"] } }] }'
```

---

## 4. Specialized Lens States (Menu Shortcuts)

### Pinhole
Sets the pinhole via the software menu shortcuts (`Alt+V` sequence).
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/pinhole
```

### Occluder (Menu Shortcut)
Sets the occluder via the software menu shortcuts (`Alt+V` sequence).
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/occluder
```

---

## 5. Reset Operations

### Global Reset (To 0/0/180)
Resets all values (SPH, CYL, AXIS) to neutral and clears occluders.
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/reset
```

---

## 6. Test Suite for State Management

These test cases demonstrate the dual-mode state management behavior.

### Test 1: Reset and Set with Correct Previous State
```bash
# Reset to 0/0/180
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/reset

# Set values with correct prev_state (0/0/180 after reset)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 1,
      "prev_aux_lens": "BINO",
      "prev_right_eye": {"sph": 0.00, "cyl": 0.00, "axis": 180},
      "prev_left_eye": {"sph": 0.00, "cyl": 0.00, "axis": 180},
      "aux_lens": "AuxLensL",
      "right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90},
      "left_eye": {"sph": -1.75, "cyl": -1.00, "axis": 180}
    }]
  }'
```
**Expected:** Should move from 0/0/180 to target values.

### Test 2: Incremental Change with Previous State
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 2,
      "prev_aux_lens": "AuxLensL",
      "prev_right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90},
      "prev_left_eye": {"sph": -1.75, "cyl": -1.00, "axis": 180},
      "aux_lens": "AuxLensR",
      "right_eye": {"sph": -3.00, "cyl": -1.50, "axis": 45},
      "left_eye": {"sph": -2.50, "cyl": -1.25, "axis": 135}
    }]
  }'
```
**Expected:** Should calculate and execute the difference.

### Test 3: No Change (prev == target)
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 3,
      "prev_aux_lens": "AuxLensR",
      "prev_right_eye": {"sph": -3.00, "cyl": -1.50, "axis": 45},
      "prev_left_eye": {"sph": -2.50, "cyl": -1.25, "axis": 135},
      "aux_lens": "AuxLensR",
      "right_eye": {"sph": -3.00, "cyl": -1.50, "axis": 45},
      "left_eye": {"sph": -2.50, "cyl": -1.25, "axis": 135}
    }]
  }'
```
**Expected:** Should skip JCC click and execute 0 eye adjustments.

### Test 4: Change Only AuxLens
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 4,
      "prev_aux_lens": "AuxLensR",
      "prev_right_eye": {"sph": -3.00, "cyl": -1.50, "axis": 45},
      "prev_left_eye": {"sph": -2.50, "cyl": -1.25, "axis": 135},
      "aux_lens": "BINO",
      "right_eye": {"sph": -3.00, "cyl": -1.50, "axis": 45},
      "left_eye": {"sph": -2.50, "cyl": -1.25, "axis": 135}
    }]
  }'
```
**Expected:** Should only click JCC BINO, no eye adjustments.

### Test 5: Without prev_state (Internal Tracking)
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 5,
      "aux_lens": "AuxLensL",
      "right_eye": {"sph": -1.00, "cyl": -0.50, "axis": 180},
      "left_eye": {"sph": -1.00, "cyl": -0.50, "axis": 180}
    }]
  }'
```
**Expected:** Uses agent's internal state tracking (may be inaccurate).

### Test 6: Partial Previous State
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "case_id": 6,
      "prev_right_eye": {"sph": -1.00, "cyl": -0.50, "axis": 180},
      "right_eye": {"sph": -2.00, "cyl": -1.00, "axis": 90},
      "left_eye": {"sph": -1.50, "cyl": -0.75, "axis": 180}
    }]
  }'
```
**Expected:** Right eye uses prev_state, left eye uses internal tracking.
