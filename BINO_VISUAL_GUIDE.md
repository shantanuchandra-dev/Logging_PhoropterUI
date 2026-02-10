# BINO Binocular Balance - Visual Guide

## Phase Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE K: BINO BALANCE                    │
│                    (Binocular Balance)                      │
└─────────────────────────────────────────────────────────────┘
```

## Chart Display

```
┌─────────────────────────────────────────────────────────────┐
│                      CHART_20 (BINO)                        │
│                                                             │
│                  ┌───────────────────┐                      │
│                  │   TOP LINE (R)    │  ← Right Eye View   │
│                  │   A B C D E F     │                      │
│                  └───────────────────┘                      │
│                                                             │
│                  ┌───────────────────┐                      │
│                  │  BOTTOM LINE (L)  │  ← Left Eye View    │
│                  │   G H I J K L     │                      │
│                  └───────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Question: "You should see 2 lines at top and bottom.
          Focus on last letter. Which one is less blurry
          than the others (if there is one)?"
```

## Decision Flow

```
                    ┌─────────────────┐
                    │  BINO Balance   │
                    │   chart_20      │
                    │   Occ: BINO     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Ask Question   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼─────────┐   │   ┌─────────▼─────────┐
    │  Top is blurry    │   │   │ Bottom is blurry  │
    │  [Right Eye]      │   │   │  [Left Eye]       │
    └─────────┬─────────┘   │   └─────────┬─────────┘
              │              │              │
    ┌─────────▼─────────┐   │   ┌─────────▼─────────┐
    │ Add 0.25D to      │   │   │ Add 0.25D to      │
    │ Left Eye SPH      │   │   │ Right Eye SPH     │
    └─────────┬─────────┘   │   └─────────┬─────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Repeat cycle   │
                    │  (ask again)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Both are same  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Test Complete!  │
                    └─────────────────┘
```

## Intent Logic Visualization

### Scenario 1: Top is Blurry

```
┌─────────────────────────────────────────────────────────────┐
│ Initial State                                               │
│ R: -1.00 / -0.50 / 90                                      │
│ L: -1.00 / -0.50 / 85                                      │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ Patient: "Top is blurry"
                             │ (Right eye sees blurry)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Action: Add 0.25D to LEFT Eye SPH                          │
│ Rationale: Compensate by strengthening the opposite eye    │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ New State                                                   │
│ R: -1.00 / -0.50 / 90  (unchanged)                         │
│ L: -0.75 / -0.50 / 85  (SPH increased by 0.25D)           │
└─────────────────────────────────────────────────────────────┘
```

### Scenario 2: Bottom is Blurry

```
┌─────────────────────────────────────────────────────────────┐
│ Initial State                                               │
│ R: -1.00 / -0.50 / 90                                      │
│ L: -1.00 / -0.50 / 85                                      │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ Patient: "Bottom is blurry"
                             │ (Left eye sees blurry)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Action: Add 0.25D to RIGHT Eye SPH                         │
│ Rationale: Compensate by strengthening the opposite eye    │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ New State                                                   │
│ R: -0.75 / -0.50 / 90  (SPH increased by 0.25D)           │
│ L: -1.00 / -0.50 / 85  (unchanged)                         │
└─────────────────────────────────────────────────────────────┘
```

### Scenario 3: Both are Same

```
┌─────────────────────────────────────────────────────────────┐
│ Current State                                               │
│ R: -0.75 / -0.50 / 90                                      │
│ L: -1.00 / -0.50 / 85                                      │
└─────────────────────────────────────────────────────────────┘
                             │
                             │ Patient: "Both are same"
                             │ (Balanced!)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│ Action: Test Complete                                       │
│ Final Prescription:                                         │
│ R: -0.75 / -0.50 / 90                                      │
│ L: -1.00 / -0.50 / 85                                      │
└─────────────────────────────────────────────────────────────┘
```

## Iterative Example

```
Round 1:
┌──────────────────────────────────────────────────────────┐
│ State: R(-1.00/-0.50/90) L(-1.50/-0.50/85)             │
│ Question: Which line is less blurry?                    │
│ Answer: Bottom is blurry [Left Eye]                     │
│ Action: Add 0.25D to Right Eye SPH                      │
│ New State: R(-0.75/-0.50/90) L(-1.50/-0.50/85)         │
└──────────────────────────────────────────────────────────┘
                        ↓
Round 2:
┌──────────────────────────────────────────────────────────┐
│ State: R(-0.75/-0.50/90) L(-1.50/-0.50/85)             │
│ Question: Which line is less blurry?                    │
│ Answer: Bottom is blurry [Left Eye]                     │
│ Action: Add 0.25D to Right Eye SPH                      │
│ New State: R(-0.50/-0.50/90) L(-1.50/-0.50/85)         │
└──────────────────────────────────────────────────────────┘
                        ↓
Round 3:
┌──────────────────────────────────────────────────────────┐
│ State: R(-0.50/-0.50/90) L(-1.50/-0.50/85)             │
│ Question: Which line is less blurry?                    │
│ Answer: Both are same                                    │
│ Action: Test Complete!                                   │
│ Final: R(-0.50/-0.50/90) L(-1.50/-0.50/85)             │
└──────────────────────────────────────────────────────────┘
```

## Previous State Feature

```
Step 1: Initial State
┌──────────────────────────────────────────────────────────┐
│ R: -1.00 / -0.50 / 90                                   │
│ L: -1.00 / -0.50 / 85                                   │
└──────────────────────────────────────────────────────────┘
                        ↓
Step 2: Patient says "Top is blurry"
┌──────────────────────────────────────────────────────────┐
│ Previous state saved:                                    │
│   R: -1.00 / -0.50 / 90                                 │
│   L: -1.00 / -0.50 / 85                                 │
│                                                          │
│ New state after adjustment:                              │
│   R: -1.00 / -0.50 / 90                                 │
│   L: -0.75 / -0.50 / 85  ← Changed                      │
│                                                          │
│ "Prev State" option now available                       │
└──────────────────────────────────────────────────────────┘
                        ↓
Step 3: Patient says "Prev State" (adjustment made it worse)
┌──────────────────────────────────────────────────────────┐
│ Restored to previous state:                              │
│   R: -1.00 / -0.50 / 90                                 │
│   L: -1.00 / -0.50 / 85                                 │
│                                                          │
│ "Prev State" option removed                             │
└──────────────────────────────────────────────────────────┘
```

## UI Display

```
┌─────────────────────────────────────────────────────────────┐
│  Phase K: Binocular Balance (Step 6.5)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Chart: bino_chart (chart_20)                              │
│  Occluder: BINO (both eyes open)                           │
│                                                             │
│  Current Power:                                             │
│    Right Eye:  -1.00 / -0.50 / 90                          │
│    Left Eye:   -0.75 / -0.50 / 85                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Question:                                                  │
│  "You should see 2 lines at top and bottom.                │
│   Focus on last letter. Which one is less blurry           │
│   than the others (if there is one)?"                      │
├─────────────────────────────────────────────────────────────┤
│  Available Responses:                                       │
│    [1] Top is blurry [Right Eye]                           │
│    [2] Bottom is blurry [Left Eye]                         │
│    [3] Both are same                                        │
│    [4] Prev State                                           │
└─────────────────────────────────────────────────────────────┘
```

## API Call Sequence

```
1. Transition to BINO Balance
   ┌────────────────────────────────────────────────────┐
   │ POST /phoropter/phoropter-1/run-tests             │
   │ {                                                  │
   │   "test_cases": [{                                 │
   │     "chart": {                                     │
   │       "tab": "Chart1",                             │
   │       "chart_items": ["chart_20"]                  │
   │     }                                              │
   │   }]                                               │
   │ }                                                  │
   └────────────────────────────────────────────────────┘

2. Set occluder to BINO
   ┌────────────────────────────────────────────────────┐
   │ POST /phoropter/phoropter-1/run-tests             │
   │ {                                                  │
   │   "test_cases": [{                                 │
   │     "right_eye": { ... },                          │
   │     "left_eye": { ... }                            │
   │   }]                                               │
   │ }                                                  │
   └────────────────────────────────────────────────────┘

3. Set JCC control to BINO
   ┌────────────────────────────────────────────────────┐
   │ POST /phoropter/phoropter-1/run-tests             │
   │ {                                                  │
   │   "test_cases": [{                                 │
   │     "jcc": "BINO"                                  │
   │   }]                                               │
   │ }                                                  │
   └────────────────────────────────────────────────────┘

4. Adjust power with previous state (Vision Correction API)
   ┌────────────────────────────────────────────────────┐
   │ POST /phoropter/phoropter-1/run-tests             │
   │ {                                                  │
   │   "test_cases": [{                                 │
   │     "case_id": 1,                                  │
   │     "prev_right_eye": {                            │
   │       "sph": -1.00, "cyl": -0.50, "axis": 90      │
   │     },                                             │
   │     "prev_left_eye": {                             │
   │       "sph": -1.00, "cyl": -0.50, "axis": 85      │
   │     },                                             │
   │     "prev_aux_lens": "BINO",                       │
   │     "right_eye": {                                 │
   │       "sph": -1.00, "cyl": -0.50, "axis": 90      │
   │     },                                             │
   │     "left_eye": {                                  │
   │       "sph": -0.75, "cyl": -0.50, "axis": 85      │
   │     },                                             │
   │     "aux_lens": "BINO"                             │
   │   }]                                               │
   │ }                                                  │
   │                                                    │
   │ Note: Uses Vision Correction API with Previous    │
   │       State and aux_lens="BINO" (both eyes open)  │
   │       for accurate click calculations              │
   └────────────────────────────────────────────────────┘
```

## State Machine Integration

```
┌─────────────────────────────────────────────────────────────┐
│                     COMPLETE FLOW                           │
└─────────────────────────────────────────────────────────────┘

Phase A: Distance Vision
         ↓
Phase B: Right Eye Refraction
         ↓
Phase E: JCC Axis Right
         ↓
Phase F: JCC Power Right
         ↓
Phase G: Duochrome Right
         ↓
Phase D: Left Eye Refraction
         ↓
Phase H: JCC Axis Left
         ↓
Phase I: JCC Power Left
         ↓
Phase J: Duochrome Left
         ↓
Phase K: BINO Balance  ← YOU ARE HERE
         ↓
    Test Complete!
```

## Summary

The BINO phase provides a final binocular balance check by:

1. **Showing chart_20** with two lines (top and bottom)
2. **Asking which line is blurry** (or if both are same)
3. **Adjusting the opposite eye** to compensate for imbalance
4. **Repeating** until both lines appear equally clear
5. **Completing the test** when balanced

This ensures both eyes work together harmoniously for optimal binocular vision.
