# JCC Large Adjustments Implementation

## Overview

This document describes the implementation of larger adjustment increments for JCC (Jackson Cross Cylinder) phases. In addition to the standard ±5° axis and ±0.25D cylinder adjustments, the system now supports "MUCH better" options for ±10° axis and ±0.50D cylinder adjustments.

## Motivation

During JCC testing, patients may sometimes have a very clear preference for one flip over the other. In these cases, making larger adjustments can speed up the refraction process while maintaining accuracy. The "MUCH better" options allow the optometrist to make faster progress when the patient's response is unambiguous.

## Implementation Details

### JCC Axis Adjustments

#### Standard Adjustments (±5°)
- **Flip 1 was better (GAP Axis)**: Increase axis by 5°
- **Flip 2 was better (RAM Axis)**: Decrease axis by 5°

#### Large Adjustments (±10°)
- **Flip 1 was MUCH better (GAP Axis)**: Increase axis by 10°
- **Flip 2 was MUCH better (RAM Axis)**: Decrease axis by 10°

**Implementation**: The 10° adjustment is achieved by calling the JCC `increase` or `decrease` operation twice (5° + 5° = 10°).

### JCC Power Adjustments

#### Standard Adjustments (±0.25D)
- **Flip 1 was better (GAP Power)**: Increase cylinder by 0.25D
- **Flip 2 was better (RAM Power)**: Decrease cylinder by 0.25D

#### Large Adjustments (±0.50D)
- **Flip 1 was MUCH better (GAP Power)**: Increase cylinder by 0.50D
- **Flip 2 was MUCH better (RAM Power)**: Decrease cylinder by 0.50D

**Implementation**: The 0.50D adjustment is achieved by calling the JCC `increase` or `decrease` operation twice (0.25D + 0.25D = 0.50D).

**Important**: Spherical equivalent compensation is tracked for each 0.25D step to ensure accurate power adjustments.

## Protocol Configuration

### JCC Axis Right (Phase E)

```yaml
intents:
  flip1: []
  flip2:
    - "Flip 1 was better (GAP Axis - increase axis by 5°)"
    - "Flip 2 was better (RAM Axis - decrease axis by 5°)"
    - "Flip 1 was MUCH better (GAP Axis - increase axis by 10°)"
    - "Flip 2 was MUCH better (RAM Axis - decrease axis by 10°)"
    - "Both Same (no change needed)"
    - "Repeat (show Flip 1 and Flip 2 again)"

adjustment_rules:
  flip1_chosen: "+5 degrees to R_AXIS"
  flip2_chosen: "-5 degrees to R_AXIS"
  flip1_much_better: "+10 degrees to R_AXIS"
  flip2_much_better: "-10 degrees to R_AXIS"
```

### JCC Power Right (Phase F)

```yaml
intents:
  flip1: []
  flip2:
    - "Flip 1 was better (GAP Power - increase cylinder by 0.25D)"
    - "Flip 2 was better (RAM Power - decrease cylinder by 0.25D)"
    - "Flip 1 was MUCH better (GAP Power - increase cylinder by 0.50D)"
    - "Flip 2 was MUCH better (RAM Power - decrease cylinder by 0.50D)"
    - "Both Same (no change needed)"
    - "Repeat (show Flip 1 and Flip 2 again)"

adjustment_rules:
  flip1_chosen: "+0.25D to R_CYL (more positive/less negative)"
  flip2_chosen: "-0.25D to R_CYL (more negative)"
  flip1_much_better: "+0.50D to R_CYL (more positive/less negative)"
  flip2_much_better: "-0.50D to R_CYL (more negative)"
```

### JCC Axis Left (Phase H)

Same structure as JCC Axis Right, but for left eye.

### JCC Power Left (Phase I)

Same structure as JCC Power Right, but for left eye.

## Code Implementation

### JCC Axis Processing

The implementation checks for "MUCH better" in the intent string before checking for standard adjustments:

```python
if "MUCH better" in intent and ("GAP Axis" in intent or "Flip 1" in intent):
    # Patient chose Flip 1 MUCH better - increase axis by 10°
    reversal = self._record_jcc_choice("flip1")
    # Call increase twice for 10° total (5° + 5°)
    self.jcc_control("increase")
    self.jcc_control("increase")
    
    # Update internal state
    self.current_row = self._copy_row_state()
    self.current_row.r_axis += 10
    if self.current_row.r_axis > 180:
        self.current_row.r_axis -= 180
    
    # Check for reversal and continue...
elif "GAP Axis" in intent or "Flip 1" in intent:
    # Standard 5° adjustment
    # ...
```

### JCC Power Processing with Spherical Equivalent

The power adjustment implementation tracks spherical equivalent compensation for each 0.25D step:

```python
if "MUCH better" in intent and ("GAP Power" in intent or "Flip 1" in intent):
    # Patient chose Flip 1 MUCH better - increase cylinder by 0.50D
    reversal = self._record_jcc_choice("flip1")
    
    # Track spherical equivalent for each 0.25D step
    self.current_row = self._copy_row_state()
    
    # First 0.25D increase
    was_at_threshold = self._is_at_cyl_threshold(self.current_row.r_cyl)
    self.jcc_control("increase")
    self.current_row.r_cyl += 0.25
    now_at_threshold = self._is_at_cyl_threshold(self.current_row.r_cyl)
    if was_at_threshold and not now_at_threshold:
        self.current_row.r_sph -= 0.25
        print(f"✓ Spherical equivalent reversion: SPH decreased by -0.25D")
    
    # Second 0.25D increase
    was_at_threshold = self._is_at_cyl_threshold(self.current_row.r_cyl)
    self.jcc_control("increase")
    self.current_row.r_cyl += 0.25
    now_at_threshold = self._is_at_cyl_threshold(self.current_row.r_cyl)
    if was_at_threshold and not now_at_threshold:
        self.current_row.r_sph -= 0.25
        print(f"✓ Spherical equivalent reversion: SPH decreased by -0.25D")
    
    # Check for reversal and continue...
```

## Usage Examples

### Example 1: JCC Axis with Large Adjustment

```
Initial State: R(-1.00/-0.50/90)

Round 1:
  Flip 1 shown → Flip 2 shown
  Patient: "Flip 1 was MUCH better"
  Action: Increase axis by 10°
  New State: R(-1.00/-0.50/100)

Round 2:
  Flip 1 shown → Flip 2 shown
  Patient: "Flip 2 was better"
  Action: Decrease axis by 5°
  New State: R(-1.00/-0.50/95)

Round 3:
  Flip 1 shown → Flip 2 shown
  Patient: "Both Same"
  → Move to JCC Power
```

### Example 2: JCC Power with Large Adjustment and Spherical Equivalent

```
Initial State: R(-1.00/-0.25/90)

Round 1:
  Flip 1 shown → Flip 2 shown
  Patient: "Flip 2 was MUCH better"
  Action: Decrease cylinder by 0.50D (two steps)
    Step 1: CYL -0.25 → -0.50 (crossed threshold, SPH +0.25)
    Step 2: CYL -0.50 → -0.75 (no threshold crossing)
  New State: R(-0.75/-0.75/90)

Round 2:
  Flip 1 shown → Flip 2 shown
  Patient: "Flip 1 was better"
  Action: Increase cylinder by 0.25D
    CYL -0.75 → -0.50 (crossed threshold, SPH -0.25)
  New State: R(-1.00/-0.50/90)

Round 3:
  Flip 1 shown → Flip 2 shown
  Patient: "Both Same"
  → Move to Duochrome
```

## Intent Ordering

The order of intent checking is critical:

1. **"Repeat"** - Check first to allow re-showing flips
2. **"MUCH better"** - Check before standard adjustments
3. **Standard adjustments** - Check after MUCH better
4. **"Both Same"** - Check last to move to next phase

This ensures that the more specific "MUCH better" intents are matched before the general "better" intents.

## Reversal Detection

The reversal detection logic works the same for both standard and large adjustments:

- If patient chooses the same flip multiple times in a row, then switches to the opposite flip, a reversal is detected
- On reversal, the phase transitions to the next phase (Axis → Power, Power → Duochrome)
- This works identically whether the patient used standard or large adjustments

## Benefits

1. **Faster Testing**: When patient has clear preference, larger adjustments speed up the process
2. **Maintains Accuracy**: Spherical equivalent compensation is tracked for each 0.25D step
3. **Flexible Workflow**: Optometrist can mix standard and large adjustments as needed
4. **Same Exit Logic**: Reversal detection and "Both Same" work identically

## UI Display

The intents are displayed in order:

```
┌─────────────────────────────────────────────────────────────┐
│  Phase E: JCC Axis Refinement (Right Eye)                  │
├─────────────────────────────────────────────────────────────┤
│  Question: Now this is Flip 2. Which was better?           │
├─────────────────────────────────────────────────────────────┤
│  Available Responses:                                       │
│    [1] Flip 1 was better (GAP Axis - increase axis by 5°) │
│    [2] Flip 2 was better (RAM Axis - decrease axis by 5°) │
│    [3] Flip 1 was MUCH better (GAP Axis - increase by 10°)│
│    [4] Flip 2 was MUCH better (RAM Axis - decrease by 10°)│
│    [5] Both Same (no change needed)                         │
│    [6] Repeat (show Flip 1 and Flip 2 again)              │
└─────────────────────────────────────────────────────────────┘
```

## Summary

The JCC large adjustments feature provides:

✅ **±10° axis adjustments** for clear axis preferences
✅ **±0.50D cylinder adjustments** for clear power preferences
✅ **Spherical equivalent tracking** for each 0.25D step
✅ **Same reversal detection** logic as standard adjustments
✅ **Flexible workflow** - mix standard and large adjustments
✅ **Consistent with existing patterns** - same code structure as standard adjustments

This enhancement speeds up the JCC testing process while maintaining the accuracy and safety of the refraction workflow.
