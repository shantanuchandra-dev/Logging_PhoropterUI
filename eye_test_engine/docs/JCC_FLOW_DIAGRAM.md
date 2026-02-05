# JCC Phase Flow Diagram

## Visual Flow Chart

```
┌─────────────────────────────────────────────────────────────────┐
│                    JCC AXIS REFINEMENT (RIGHT EYE)              │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │   FLIP 1 SHOWN       │
    │   (Right_Axis_Flip1) │
    │                      │
    │   No Intent Buttons  │
    │   Countdown: 2s      │
    └──────────┬───────────┘
               │ (automatic after 2s)
               │ AUTO_FLIP intent sent
               ↓
    ┌──────────────────────┐
    │   FLIP 2 SHOWN       │
    │   (Right_Axis_Flip2) │
    │                      │
    │   4 Intent Buttons:  │
    │   1. Flip 1 better   │ ──→ Increase axis +5°  ──┐
    │   2. Flip 2 better   │ ──→ Decrease axis -5°  ──┤
    │   3. Both Same       │ ──→ Move to Power      ──┤
    │   4. Repeat          │ ──→ Back to Flip 1     ──┤
    └──────────────────────┘                          │
               │                                      │
               │ (if Flip 1 or Flip 2 selected)     │
               ↓                                      │
    ┌──────────────────────┐                         │
    │   ADJUSTMENT MADE    │                         │
    │   Axis changed       │                         │
    │   Back to FLIP 1     │ ←───────────────────────┘
    │   Countdown: 2s      │
    └──────────┬───────────┘
               │
               ↓ (cycle repeats)
               
    ┌──────────────────────┐
    │   "BOTH SAME"        │
    │   selected           │
    └──────────┬───────────┘
               │ (immediate transition)
               ↓
    ┌──────────────────────┐
    │   JCC POWER          │
    │   REFINEMENT         │
    │   (Right_Power_Flip1)│
    └──────────────────────┘
```

---

## Timing Diagram

```
TIME    STATE              DISPLAY                    USER ACTION
─────────────────────────────────────────────────────────────────
t=0s    Flip1             "Focus on dot chart..."    (observing)
        Right_Axis_Flip1  Countdown: 2s
                          No buttons

t=2s    AUTO_FLIP sent    (processing...)            (none)
        
t=2.1s  Flip2             "Now this is Flip 2..."    [4 buttons appear]
        Right_Axis_Flip2  - Flip 1 better
                          - Flip 2 better
                          - Both Same
                          - Repeat

t=5s    (user clicks)     Processing...              Click "Flip 1 better"

t=5.1s  Flip1             Axis adjusted +5°          (observing)
        Right_Axis_Flip1  Countdown: 2s
                          No buttons

t=7.1s  AUTO_FLIP sent    (processing...)            (none)

t=7.2s  Flip2             "Now this is Flip 2..."    [4 buttons appear]
        Right_Axis_Flip2  

... cycle continues until "Both Same" selected ...

t=15s   (user clicks)     Processing...              Click "Both Same"

t=15.1s Power Flip1       "Focus on dot chart..."    (observing)
        Right_Power_Flip1 Countdown: 2s
                          [Transitioned to Power phase]
```

---

## State Transitions

### Normal Adjustment Flow
```
Flip1 (auto_flip=true)
  ↓ (2s countdown)
  ↓ AUTO_FLIP
Flip2 (intents=[...])
  ↓ User selects "Flip 1 better"
  ↓ Adjust axis +5°
Flip1 (auto_flip=true)
  ↓ (cycle repeats)
```

### Repeat Flow
```
Flip2 (intents=[...])
  ↓ User selects "Repeat"
Flip1 (auto_flip=true)
  ↓ (2s countdown)
  ↓ AUTO_FLIP
Flip2 (intents=[...])
  ↓ (shows same flips again)
```

### Exit Flow
```
Flip2 (intents=[...])
  ↓ User selects "Both Same"
Next Phase (immediate)
  ↓ (Axis → Power → Duochrome)
```

---

## Backend Response Structure

### Flip 1 Response
```json
{
  "phase": "jcc_axis_right",
  "question": "Focus on the dot chart. This is Flip 1...",
  "intents": [],
  "auto_flip": true,
  "flip_wait_seconds": 2,
  "chart": "jcc_chart",
  "occluder": "Right_Axis_Flip1",
  "power": { ... }
}
```

### Flip 2 Response (after AUTO_FLIP)
```json
{
  "phase": "jcc_axis_right",
  "question": "Now this is Flip 2. Which was better?",
  "intents": [
    "Flip 1 was better (GAP Axis - increase axis by 5°)",
    "Flip 2 was better (RAM Axis - decrease axis by 5°)",
    "Both Same (no change needed)",
    "Repeat (show Flip 1 and Flip 2 again)"
  ],
  "chart": "jcc_chart",
  "occluder": "Right_Axis_Flip2",
  "power": { ... }
}
```

### After User Selection (back to Flip 1)
```json
{
  "phase": "jcc_axis_right",
  "question": "Focus on the dot chart. This is Flip 1...",
  "intents": [],
  "auto_flip": true,
  "flip_wait_seconds": 2,
  "chart": "jcc_chart",
  "occluder": "Right_Axis_Flip1",
  "power": {
    "right": {
      "axis": 95  // Changed from 90
    }
  }
}
```

---

## Complete JCC Sequence

```
START: Distance Vision Complete
  ↓
┌─────────────────────────────────┐
│ RIGHT EYE REFRACTION            │
│ (Left_Occluded + Snellen)       │
└───────────────┬─────────────────┘
                ↓
┌─────────────────────────────────┐
│ JCC AXIS RIGHT                  │
│ Flip1 ↔ Flip2 (cycle)          │
│ Until "Both Same"               │
└───────────────┬─────────────────┘
                ↓
┌─────────────────────────────────┐
│ JCC POWER RIGHT                 │
│ Flip1 ↔ Flip2 (cycle)          │
│ Until "Both Same"               │
└───────────────┬─────────────────┘
                ↓
┌─────────────────────────────────┐
│ DUOCHROME RIGHT                 │
│ Red vs Green                    │
└───────────────┬─────────────────┘
                ↓
┌─────────────────────────────────┐
│ LEFT EYE REFRACTION             │
│ (Right_Occluded + Snellen)      │
└───────────────┬─────────────────┘
                ↓
┌─────────────────────────────────┐
│ JCC AXIS LEFT                   │
│ Flip1 ↔ Flip2 (cycle)          │
└───────────────┬─────────────────┘
                ↓
┌─────────────────────────────────┐
│ JCC POWER LEFT                  │
│ Flip1 ↔ Flip2 (cycle)          │
└───────────────┬─────────────────┘
                ↓
┌─────────────────────────────────┐
│ DUOCHROME LEFT                  │
│ Red vs Green                    │
└───────────────┬─────────────────┘
                ↓
┌─────────────────────────────────┐
│ BINOCULAR BALANCE               │
│ (BINO + Snellen)                │
└─────────────────────────────────┘
```

---

## Key Points

1. **Flip 1**: Always shows for 2 seconds with no buttons (observation only)
2. **AUTO_FLIP**: Automatic transition after 2 seconds (no user input)
3. **Flip 2**: Shows with 4 intent buttons (user makes selection)
4. **Immediate Transitions**: All state changes happen immediately with no delays
5. **Repeat**: Goes back to Flip 1 and restarts the countdown automatically
6. **Both Same**: Exits to next phase immediately (Axis → Power → Duochrome)
