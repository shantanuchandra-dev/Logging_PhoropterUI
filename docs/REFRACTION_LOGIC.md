# Refraction Logic — Complete Documentation

## Overview

This document covers the complete refraction logic implementation, including chart progression, power adjustments, state machine flow, occlusion mechanism, pinhole testing, duochrome reversal, and curation/optotype logic.

---

## Chart Progression

### Snellen Chart Order (Big → Small)

```
snellen_chart_200_150  ← Start here (largest)
snellen_chart_100_80
snellen_chart_70_60_50
snellen_chart_40_30_25
snellen_chart_25_20_15
snellen_chart_20_20_20  ← Target
snellen_chart_20_15_10  ← Smallest
```

### Rules

| Intent | Action |
|--------|--------|
| **Able to read** | Move to next smaller chart |
| **Blurry** | Add -0.25D SPH, stay on same chart |
| **Unable to read** | Add -0.25D SPH, increment counter |

**Exit to JCC:** 2 consecutive "Unable to read" with SPH changes, OR patient successfully reads 20/20/20.

---

## Power Adjustment Values

| Parameter | Step |
|-----------|------|
| Sphere (SPH) | -0.25D per step |
| Cylinder (CYL) | ±0.25D (JCC GAP/RAM) |
| Axis | ±5° (JCC GAP/RAM) |

---

## State Machine Diagram

```
Phase A: Distance Vision (BINO)
    ↓ baseline established
    ↓ [A4 Pinhole if unable to read 20/20]
Phase B: Right Eye Refraction (Left_Occluded)
    ↓ [2x "Unable to read" after SPH adjustments]
Phase E: JCC Axis Right  → Flip1 ↔ Flip2 until stable
    ↓
Phase F: JCC Power Right → Flip1 ↔ Flip2 until stable
    ↓ [skip if CYL = 0.00]
Phase G: Duochrome Right (Left_Occluded)
    ↓
Phase D: Left Eye Refraction (Right_Occluded)
    ↓ [same 2x "Unable to read" trigger]
Phase H: JCC Axis Left
    ↓
Phase I: JCC Power Left
    ↓
Phase J: Duochrome Left
    ↓
Phase K: Binocular Balance (BINO)
    ↓
Test Complete
```

### Phase Transition Rules

| Transition | Condition |
|-----------|-----------|
| A → B | Baseline distance VA established |
| B → E | 2x "Unable to read" after SPH changes |
| E → F | Axis stable (Both Same) |
| F → G | Power stable, Both Same, or 0.00 CYL |
| G → D | Any duochrome response received |
| D → H | 2x "Unable to read" (same as B→E) |
| H → I | Axis stable |
| I → J | Power stable or 0.00 CYL |
| J → K | Any duochrome response |
| K → End | "Both are same" in binocular balance |

---

## Example Flow

### Right Eye Refraction

```
1. snellen_chart_200_150, R_SPH=0.0 → "Able to read"
2. snellen_chart_100_80, R_SPH=0.0  → "Blurry"
3. snellen_chart_100_80, R_SPH=-0.25 → "Able to read"
4. snellen_chart_70_60_50, R_SPH=-0.25 → "Able to read"
5. snellen_chart_40_30_25, R_SPH=-0.25 → "Unable to read" (count=1)
6. snellen_chart_40_30_25, R_SPH=-0.50 → "Unable to read" (count=2)
7. → Exit to JCC Axis Right
```

### JCC Axis Example

```
1. Flip1: R_AXIS=180°  (no response expected)
2. Flip2: R_AXIS=180°  → "Flip 1: GAP Axis"
3. R_AXIS=185° (wrapped: 5°)  → Flip1 again
4. Flip2: R_AXIS=5°   → "Both Same"
5. → Exit to JCC Power
```

---

## Occlusion Mechanism

### How Occlusion Works

Occlusion is handled via the **JCC eye mode** setting, not explicit `aux_lens` commands. When `set_power(occluder="Left_Occluded")` is called, it maps to `jcc_control("R")` which automatically activates the phoropter's auxiliary lens.

```python
# Inside set_power()
if occluder == "Left_Occluded":
    jcc_eye_mode = "R"   # Test right eye when left is occluded
elif occluder == "Right_Occluded":
    jcc_eye_mode = "L"   # Test left eye when right is occluded
elif occluder == "BINO":
    jcc_eye_mode = "BINO"

# Set JCC eye mode for non-JCC phases
if jcc_eye_mode and not is_jcc_phase:
    self.jcc_control(jcc_eye_mode)
```

### Occluder Mapping

| Occluder State | JCC Eye Mode | Effect |
|----------------|--------------|--------|
| `Left_Occluded` | `"R"` | Left eye occluded via auxiliary lens |
| `Right_Occluded` | `"L"` | Right eye occluded via auxiliary lens |
| `BINO` | `"BINO"` | Both eyes open |

### CURL Command Sent (Phase A → Phase B)

```bash
# 1. Display chart
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_10"]}}]}'

# 2. Occlude left eye via JCC mode
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"jcc": "R"}]}'
```

---

## Pinhole Test During Distance Vision

### When It Triggers

Phase A (Distance Vision) → patient selects **"Unable to read"** on the E-chart.

### Workflow

```
1. Patient views echart_400
2. Patient clicks "Unable to read"
3. System activates pinhole:
   curl -X POST .../phoropter/{PHOROPTER_ID}/pinhole
4. Question: "With pinhole: Can you see the E clearly now?"
5. Patient selects:
   - "Able to read with pinhole" → likely refractive error
   - "Still unable to read"      → may indicate pathology
6. Either way: proceed to Phase B
```

### Clinical Significance

| Pinhole Result | Interpretation |
|---------------|----------------|
| Vision improves | Refractive error — correctable with lenses |
| Vision unchanged | Possible pathology (cataracts, amblyopia, etc.) — flag for further evaluation |

### Implementation

```python
def set_pinhole(self):
    cmd = f"curl -X POST {self.base_url}/phoropter/{self.phoropter_id}/pinhole"
    subprocess.run(cmd, shell=True, capture_output=True)

def _process_distance_vision(self, intent: str) -> Dict:
    if intent == "Unable to read":
        self.set_pinhole()
        response = self._build_response()
        response['question'] = "With pinhole: Can you see the E clearly now?"
        response['intents'] = ["Able to read with pinhole", "Still unable to read"]
        return response
    elif intent in ["Able to read with pinhole", "Still unable to read"]:
        return self._transition_to_right_eye_refraction()
```

---

## Duochrome Reversal Power Update Fix

### Problem

When a reversal occurred during Duochrome (e.g., Green → Red), the power was updated internally and the CURL was sent, but the **frontend did not display the updated power value**.

### Root Cause

The `_transition_to_left_eye_refraction()` function intentionally removes the `power` key from its response (to avoid redundant `setPower` calls on normal transitions). But on reversal, power **has** changed, so the frontend needs it.

### Fix

```python
# _process_duochrome_right()
if reversal:
    response = self._transition_to_left_eye_refraction()
    response['power'] = self._build_response()['power']  # Re-add updated power
    return response

# _process_duochrome_left()
if reversal:
    response = self._transition_to_binocular_balance()
    if 'power' not in response:
        response['power'] = self._build_response()['power']
    return response
```

### Test Result

```
1. Choosing Green... SPH: -1.00 → -0.75
2. Choosing Red (reversal)... SPH: -0.75 → -1.00
   Power in response: Yes  ✓ SUCCESS
```

---

## "Getting Better" — Commented Out

The "Getting better" intent has been removed from both refraction phases. It was creating confusion in clinical workflows.

### Current Available Options (Refraction Phases)

1. Able to read
2. Blurry
3. Unable to read
4. Prev State *(appears after "Blurry")*

### How to Re-enable

```yaml
# protocol.yaml — uncomment:
# - "Getting better"
```

```python
# interactive_session.py — uncomment the elif block for "Getting better"
```

---

## Curation Logic — Snellen & JCC Charts

### Snellen Parsing

Chart names like `snellen_chart_40_30_25_40` are parsed into:
- **base**: `snellen_chart_40_30_25`
- **highlight**: `40` (the currently displayed optotype size)

**Metric-to-Imperial Conversion** (all charts normalized to 20/x system):

| Metric | Imperial |
|--------|----------|
| 6/60 | 20/200 |
| 6/30 | 20/100 |
| 6/20 | 20/70 |
| 6/15 | 20/50 |
| 6/12 | 20/40 |
| 6/9 | 20/30 |
| 6/7.5 | 20/25 |
| 6/6 | 20/20 |
| 6/5 | 20/16 |
| 6/4 | 20/13 |
| 6/3 | 20/10 |

### Snellen Decision Priority (top → bottom)

1. If no previous snellen row: highlight == 20 → "Able to read." else "Blurry."
2. Same base, same highlight:
   - SPH changed → "Getting better."
   - SPH unchanged → "Blurry."
   - SPH oscillation across 3 rows → "Unable to read."
3. Same base, finer line (smaller highlight) → "Able to read."
4. Same base, coarser line (larger highlight) → "Unable to read."
5. Cross-base lookahead: next row is Snellen with finer highlight → "Able to read."
6. Fallback: highlight == 20 → "Able to read." else "Blurry."

### JCC Intent Labels

- **Flip 1 (increase movement)** → `Flip 1 - GAP - Green Add Plus`
- **Flip 2 (decrease movement)** → `Flip 2 - RAM - Red Add Minus`

Examples:
- `Flip 2 - RAM - Red Add Minus (axis decreased: 175 → 180)`
- `Flip 1 - GAP - Green Add Plus (power increased: -0.25 → -0.00)`

---

## Optotype Sizes Reference

### Understanding Snellen Fractions

A Snellen fraction (e.g., 20/20) means: patient reads at 20 feet what a person with normal vision reads at 20 feet.

### Standard Letter Heights at 6m Testing Distance

| Snellen | Denominator | Letter Height |
|---------|-------------|---------------|
| 6/60 (20/200) | 60 | 87.27 mm |
| 6/30 (20/100) | 30 | 43.63 mm |
| 6/20 (20/70) | 20 | 29.09 mm |
| 6/12 (20/40) | 12 | 17.45 mm |
| 6/9 (20/30) | 9 | 13.09 mm |
| 6/6 (20/20) | 6 | 8.73 mm |
| 6/3 (20/10) | 3 | 4.36 mm |

**Formula:** $H = d \times \tan(5') \approx d \times 0.001454$ (where $d$ is denominator distance in mm)

### Color Bar Significance

- **Red Line**: ~6/9 (20/30) — minimum for driving in many jurisdictions
- **Green Line**: ~6/6 (20/20) — "normal" vision target

---

## Phase Processing Methods Reference

```python
_process_distance_vision()
_process_right_eye_refraction()
_process_jcc_axis_right()
_process_jcc_power_right()
_process_duochrome_right()
_process_left_eye_refraction()
_process_jcc_axis_left()
_process_jcc_power_left()
_process_duochrome_left()
_process_binocular_balance()
```

---

## Files Modified

- `eye_test_engine/interactive_session.py` — All phase processing methods, duochrome fix, pinhole, "getting better" commented out
- `eye_test_engine/config/protocol.yaml` — "Getting better" commented out, JCC intents updated
- `curate_conversations.py` — Curation logic implementation
- `docs/optotype_sizes.md` — Reference for optotype size mapping

**Status:** ✅ Complete and clinically accurate
