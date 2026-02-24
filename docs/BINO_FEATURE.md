# BINO Binocular Balance Feature

## Overview

Phase K (Binocular Balance) is the final phase of the refraction test. Both eyes are open and the patient views a split chart (chart_20) to verify that both eyes are equally balanced before prescribing.

| Property | Value |
|----------|-------|
| **Phase ID** | K |
| **Phase Name** | Binocular Balance (Step 6.5) |
| **Chart** | chart_20 (bino_chart) |
| **Occluder** | BINO (both eyes open) |
| **Triggered after** | Left Eye Duochrome (Phase J) |

---

## How It Works

The patient sees two lines on chart_20 — a top line (right eye dominant) and a bottom line (left eye dominant):

```
┌─────────────────────────────────────────────────────────────┐
│                  ┌───────────────────┐                      │
│                  │   TOP LINE (R)    │  ← Right Eye View   │
│                  │   A B C D E F     │                      │
│                  └───────────────────┘                      │
│                  ┌───────────────────┐                      │
│                  │  BOTTOM LINE (L)  │  ← Left Eye View    │
│                  │   G H I J K L     │                      │
│                  └───────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

**Question:** "You should see 2 lines at top and bottom. Focus on last letter. Which one is less blurry than the others (if there is one)?"

---

## Intent Logic

| Intent | Action | Rationale |
|--------|--------|-----------|
| **Top is blurry [Right Eye]** | Add +0.25D to **Left** Eye SPH | Compensate opposite eye to re-balance |
| **Bottom is blurry [Left Eye]** | Add +0.25D to **Right** Eye SPH | Compensate opposite eye to re-balance |
| **Both are same** | Test complete | Balance achieved |
| **Prev State** | Restore previous power | Undo last adjustment |

### Decision Flow

```
┌────────────────────┐
│  Show chart_20     │
│  BINO occluder     │
└────────┬───────────┘
         │
    ┌────▼────────────────────────────────────┐
    │ Which line is less blurry?              │
    ├──────────────┬──────────────────────────┤
    │ Top blurry   │ Bottom blurry            │
    │ +0.25D L_SPH │ +0.25D R_SPH            │
    └──────┬───────┴────────────┬─────────────┘
           │                   │
           └────────┬──────────┘
                    │ Repeat until "Both are same"
                    ▼
             Test Complete!
```

---

## Example Flow

```
Initial:  R(-1.00/-0.50/90)  L(-1.50/-0.50/85)

Round 1: "Bottom is blurry" → +0.25D R_SPH
         R(-0.75/-0.50/90)  L(-1.50/-0.50/85)

Round 2: "Bottom is blurry" → +0.25D R_SPH
         R(-0.50/-0.50/90)  L(-1.50/-0.50/85)

Round 3: "Both are same" → Test Complete!
         Final: R(-0.50/-0.50/90)  L(-1.50/-0.50/85)
```

---

## Previous State Support

Before each adjustment, the current state is saved so the patient can undo with "Prev State":

```
Step 1: R(-1.00/-0.50/90) L(-1.00/-0.50/85)
        Patient: "Top is blurry"

Step 2: Saved previous state
        → L_SPH adjusted: L(-0.75/-0.50/85)
        → "Prev State" option appears

Step 3: Patient: "Prev State" (adjustment made it worse)
        → Restored: L(-1.00/-0.50/85)
        → "Prev State" removed
```

---

## Power Adjustment with Previous State

All adjustments in BINO phase use the **Vision Correction API with Previous State** for accurate click calculations:

```bash
# Example: Top is blurry → add 0.25D to Left Eye
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{
    "test_cases": [{
      "case_id": 1,
      "prev_right_eye": {"sph": -1.00, "cyl": -0.50, "axis": 90},
      "prev_left_eye": {"sph": -1.00, "cyl": -0.50, "axis": 85},
      "prev_aux_lens": "BINO",
      "right_eye": {"sph": -1.00, "cyl": -0.50, "axis": 90},
      "left_eye": {"sph": -0.75, "cyl": -0.50, "axis": 85},
      "aux_lens": "BINO"
    }]
  }'
```

```bash
# Example: Bottom is blurry → add 0.25D to Right Eye
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{
    "test_cases": [{
      "prev_right_eye": {"sph": -1.00, "cyl": -0.50, "axis": 90},
      "prev_left_eye": {"sph": -1.00, "cyl": -0.50, "axis": 85},
      "prev_aux_lens": "BINO",
      "right_eye": {"sph": -0.75, "cyl": -0.50, "axis": 90},
      "left_eye": {"sph": -1.00, "cyl": -0.50, "axis": 85},
      "aux_lens": "BINO"
    }]
  }'
```

`aux_lens` is always `"BINO"` in this phase.

---

## Implementation

### Transition Method

```python
def _transition_to_binocular_balance(self) -> Dict:
    self.current_phase = "binocular_balance"
    self.previous_state = None
    self.show_prev_state_option = False

    self.current_row.occluder_state = "BINO"
    self.current_row.chart_display = "bino_chart"

    self.set_chart("bino_chart")
    self.set_power(occluder="BINO")
    self.jcc_control("BINO")

    return self._build_response()
```

### Processing Method

```python
def _process_binocular_balance(self, intent: str) -> Dict:
    if intent == "Top is blurry [Right Eye]":
        # Save state, then add +0.25D to L_SPH
        self._save_previous_state()
        new_l_sph = self.current_row.l_sph + 0.25
        self.set_power_with_prev_state(..., l_sph=new_l_sph, aux_lens="BINO")
        self.show_prev_state_option = True
        return self._build_response()

    elif intent == "Bottom is blurry [Left Eye]":
        # Save state, then add +0.25D to R_SPH
        self._save_previous_state()
        new_r_sph = self.current_row.r_sph + 0.25
        self.set_power_with_prev_state(..., r_sph=new_r_sph, aux_lens="BINO")
        self.show_prev_state_option = True
        return self._build_response()

    elif intent == "Both are same":
        return {"status": "complete", "final_prescription": ...}

    elif intent == "Prev State":
        self._restore_previous_state()
        return self._build_response()
```

### Protocol Configuration

```yaml
binocular_balance:
  id: "K"
  name: "Binocular Balance (Step 6.5)"
  charts: ["bino_chart"]
  occluder: "BINO"
  question: "You should see 2 lines at top and bottom. Focus on last letter. Which one is less blurry than the others (if there is one)?"
  intents:
    - "Top is blurry [Right Eye]"
    - "Bottom is blurry [Left Eye]"
    - "Both are same"
  adjustment_rules:
    top_blurry: "+0.25D to L_SPH"
    bottom_blurry: "+0.25D to R_SPH"
```

---

## Chart Display Command

```bash
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_20"]}}]}'
```

---

## Testing

Test file: `eye_test_engine/tests/test_binocular_balance_logic.py`

| Test | Verifies |
|------|---------|
| `test_binocular_balance_top_blurry` | "Top is blurry" adds +0.25D to L_SPH |
| `test_binocular_balance_bottom_blurry` | "Bottom is blurry" adds +0.25D to R_SPH |
| `test_binocular_balance_both_same` | "Both are same" completes the test |
| `test_binocular_balance_prev_state` | "Prev State" restores previous power |
| `test_binocular_balance_iterative` | Multiple rounds until balanced |

All tests pass ✅

---

## Files Modified

- `eye_test_engine/interactive_session.py` — `_transition_to_binocular_balance()`, `_process_binocular_balance()`
- `eye_test_engine/config/protocol.yaml` — Phase K configuration

**Status:** ✅ Complete — Vision Correction API with Previous State used for all adjustments
