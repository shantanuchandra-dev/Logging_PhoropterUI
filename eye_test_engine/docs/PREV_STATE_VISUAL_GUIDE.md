# Visual Guide: Prev State Feature

## Scenario: Right Eye Refraction Test

### Step 1: Initial State
```
┌─────────────────────────────────────────────────────────┐
│ Phase: Phase B: Right Eye Refraction (Step 6.1)       │
├─────────────────────────────────────────────────────────┤
│ Question: I'm covering your left eye.                  │
│           Please read the line you can see clearly.    │
├─────────────────────────────────────────────────────────┤
│ Chart: snellen_chart_20_20_20                          │
│ Power: Right Eye: 0.00 / 0.00 / 180°                   │
│        Left Eye:  0.00 / 0.00 / 180°                   │
├─────────────────────────────────────────────────────────┤
│ Available Options:                                      │
│   1. Able to read                                       │
│   2. Blurry                           ← User clicks this│
│   3. Unable to read                                     │
│   4. Getting better                                     │
└─────────────────────────────────────────────────────────┘
```

### Step 2: After Clicking "Blurry"
```
┌─────────────────────────────────────────────────────────┐
│ Phase: Phase B: Right Eye Refraction (Step 6.1)       │
├─────────────────────────────────────────────────────────┤
│ Question: I'm covering your left eye.                  │
│           Please read the line you can see clearly.    │
├─────────────────────────────────────────────────────────┤
│ Chart: snellen_chart_20_20_20                          │
│ Power: Right Eye: -0.25 / 0.00 / 180°  ← Changed!      │
│        Left Eye:  0.00 / 0.00 / 180°                   │
├─────────────────────────────────────────────────────────┤
│ Available Options:                                      │
│   1. Able to read                                       │
│   2. Blurry                                             │
│   3. Unable to read                                     │
│   4. Getting better                                     │
│   5. Prev State                       ← NEW OPTION!     │
└─────────────────────────────────────────────────────────┘

Backend: Saved previous state (R SPH: 0.00)
Phoropter Command Sent:
  curl -X POST .../run-tests \
    -d '{"test_cases": [{"right_eye": {"sph": -0.25}}]}'
```

### Step 3: User Clicks "Prev State"
```
┌─────────────────────────────────────────────────────────┐
│ Phase: Phase B: Right Eye Refraction (Step 6.1)       │
├─────────────────────────────────────────────────────────┤
│ Question: I'm covering your left eye.                  │
│           Please read the line you can see clearly.    │
├─────────────────────────────────────────────────────────┤
│ Chart: snellen_chart_20_20_20                          │
│ Power: Right Eye: 0.00 / 0.00 / 180°   ← Restored!     │
│        Left Eye:  0.00 / 0.00 / 180°                   │
├─────────────────────────────────────────────────────────┤
│ Available Options:                                      │
│   1. Able to read                                       │
│   2. Blurry                                             │
│   3. Unable to read                                     │
│   4. Getting better                                     │
│   (Prev State removed)                                  │
└─────────────────────────────────────────────────────────┘

Backend: Restored previous state
Phoropter Command Sent:
  curl -X POST .../run-tests \
    -d '{"test_cases": [{"right_eye": {"sph": 0.0}}]}'
```

## Alternative Flow: Clicking Another Option

### After "Blurry" → Click "Able to read"
```
┌─────────────────────────────────────────────────────────┐
│ Phase: Phase B: Right Eye Refraction (Step 6.1)       │
├─────────────────────────────────────────────────────────┤
│ Question: I'm covering your left eye.                  │
│           Please read the line you can see clearly.    │
├─────────────────────────────────────────────────────────┤
│ Chart: snellen_chart_100_80              ← New chart    │
│ Power: Right Eye: -0.25 / 0.00 / 180°   ← Kept         │
│        Left Eye:  0.00 / 0.00 / 180°                   │
├─────────────────────────────────────────────────────────┤
│ Available Options:                                      │
│   1. Able to read                                       │
│   2. Blurry                                             │
│   3. Unable to read                                     │
│   4. Getting better                                     │
│   (Prev State removed - user accepted the change)      │
└─────────────────────────────────────────────────────────┘
```

## Left Eye Example

### Step 1: Initial State (Left Eye)
```
┌─────────────────────────────────────────────────────────┐
│ Phase: Phase D: Left Eye Refraction (Step 6.3)        │
├─────────────────────────────────────────────────────────┤
│ Question: I'm covering your right eye.                 │
│           Please read the line you can see clearly.    │
├─────────────────────────────────────────────────────────┤
│ Chart: snellen_chart_20_20_20                          │
│ Power: Right Eye: -0.50 / -0.25 / 175°                 │
│        Left Eye:  0.00 / 0.00 / 180°                   │
├─────────────────────────────────────────────────────────┤
│ Available Options:                                      │
│   1. Able to read                                       │
│   2. Blurry                           ← User clicks this│
│   3. Unable to read                                     │
│   4. Getting better                                     │
└─────────────────────────────────────────────────────────┘
```

### Step 2: After "Blurry" (Left Eye)
```
┌─────────────────────────────────────────────────────────┐
│ Phase: Phase D: Left Eye Refraction (Step 6.3)        │
├─────────────────────────────────────────────────────────┤
│ Question: I'm covering your right eye.                 │
│           Please read the line you can see clearly.    │
├─────────────────────────────────────────────────────────┤
│ Chart: snellen_chart_20_20_20                          │
│ Power: Right Eye: -0.50 / -0.25 / 175°                 │
│        Left Eye:  -0.25 / 0.00 / 180°  ← Changed!      │
├─────────────────────────────────────────────────────────┤
│ Available Options:                                      │
│   1. Able to read                                       │
│   2. Blurry                                             │
│   3. Unable to read                                     │
│   4. Getting better                                     │
│   5. Prev State                       ← NEW OPTION!     │
└─────────────────────────────────────────────────────────┘

Phoropter Command Sent:
  curl -X POST .../run-tests \
    -d '{"test_cases": [{"left_eye": {"sph": -0.25}}]}'
```

## Key Benefits

✅ **User-Friendly**: Allows users to undo accidental "Blurry" clicks
✅ **Safe**: Only one level of undo to prevent confusion
✅ **Clear**: Option only appears when applicable
✅ **Automatic**: Frontend integration requires no changes
✅ **Consistent**: Works identically for both right and left eye tests

## Implementation Notes

- The "Prev State" button appears immediately after clicking "Blurry"
- It sends the exact same CURL command format as the initial power setting
- The option disappears after being used or when another option is selected
- The saved state includes full power parameters for accuracy
