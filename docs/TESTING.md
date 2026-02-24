# Testing Documentation

## Overview

All test files are located in `eye_test_engine/tests/`. Tests verify clinical logic, state management, power calculations, and API correctness without requiring the backend server to be running.

---

## Test File Locations

```
eye_test_engine/
└── tests/
    ├── test_prev_state.py              # Prev State for "Blurry"
    ├── test_prev_state_unable_read.py  # Prev State for "Unable to read"
    ├── test_no_getting_better.py       # "Getting better" removed
    ├── test_jcc_power_zero.py          # JCC Power zero-cylinder logic
    ├── test_binocular_balance_logic.py # BINO phase logic
    ├── test_distance_vision_chart_selector.py  # Chart selector
    ├── test_duochrome_simple.py        # Duochrome reversal fix
    ├── test_pinhole.py                 # Pinhole test feature
    └── demo_prev_state.py              # Interactive demo
```

All tests use relative imports:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from interactive_session import InteractiveSession
```

---

## Available Tests

### 1. `test_prev_state.py`

**What it tests:**
- ✓ "Prev State" appears after "Blurry" (right eye)
- ✓ Power restored correctly after "Prev State"
- ✓ Same for left eye
- ✓ "Prev State" does NOT appear for other intents

```bash
cd eye_test_engine/tests && python test_prev_state.py
```

### 2. `test_prev_state_unable_read.py`

**What it tests:**
- ✓ "Prev State" appears after "Unable to read"
- ✓ "Prev State" still appears after "Blurry"
- ✓ Power is correctly restored when "Prev State" is clicked

```bash
cd eye_test_engine/tests && python test_prev_state_unable_read.py
```

### 3. `test_no_getting_better.py`

**What it tests:**
- ✓ "Getting better" NOT in intents for right eye refraction
- ✓ "Getting better" NOT in intents for left eye refraction

```bash
cd eye_test_engine/tests && python test_no_getting_better.py
```

### 4. `test_jcc_power_zero.py`

**What it tests:**
- ✓ Right eye: First "Flip 1 at 0.00 CYL" repeats flip cycle
- ✓ Right eye: Second occurrence moves to duochrome
- ✓ Same for left eye

```bash
cd eye_test_engine/tests && python test_jcc_power_zero.py
```

### 5. `test_binocular_balance_logic.py`

**What it tests:**
- ✓ "Top is blurry" adds +0.25D to L_SPH
- ✓ "Bottom is blurry" adds +0.25D to R_SPH
- ✓ "Both are same" completes test
- ✓ "Prev State" restores power
- ✓ Iterative adjustments until balanced

### 6. `test_distance_vision_chart_selector.py`

**What it tests:**
- ✓ `chart_info` present in distance vision response
- ✓ Chart switching works correctly
- ✓ Chart selector persists through pinhole test

### 7. `test_duochrome_simple.py`

**What it tests:**
- ✓ Power included in response after duochrome reversal
- ✓ Power value matches updated internal state

### 8. `test_pinhole.py`

**What it tests:**
- ✓ Pinhole is activated when patient can't read E-chart
- ✓ Vision improves with pinhole → proceeds to refraction
- ✓ Vision still poor with pinhole → still proceeds to refraction

---

## Running All Tests

```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests

python test_prev_state.py && \
python test_prev_state_unable_read.py && \
python test_no_getting_better.py && \
python test_jcc_power_zero.py

echo "All tests completed!"
```

---

## Test Results Summary

| Test File | Status |
|-----------|--------|
| `test_prev_state.py` | ✅ Passing |
| `test_no_getting_better.py` | ✅ Passing |
| `test_jcc_power_zero.py` | ✅ Passing |
| `test_binocular_balance_logic.py` | ✅ Passing |
| `test_prev_state_unable_read.py` | ✅ Ready |
| `test_distance_vision_chart_selector.py` | ✅ Passing |

---

## JCC Auto-Flip Testing Checklist

Use this checklist to manually verify the JCC auto-flip implementation end-to-end:

### Phase A: Distance Vision
- [ ] Button becomes disabled on click
- [ ] Phoropter reset happens **once**
- [ ] E-chart 400 displayed
- [ ] Intents: Able to read, Blurry, Unable to read

### Phase B: Right Eye Refraction
- [ ] Starts with **snellen_chart_200_150** (largest)
- [ ] Occluder: Left_Occluded
- [ ] Progress through charts on "Able to read"

### Phase E: JCC Axis Right (Auto-Flip)
- [ ] JCC chart displayed **once**
- [ ] Question: "This is Flip 1. (Flip 2 will show automatically in 2 seconds)"
- [ ] **NO intent buttons** on Flip 1
- [ ] Countdown: "⏱️ Showing Flip 2 in 2 seconds..."
- [ ] After 2 seconds: `jcc: "handle"` sent
- [ ] Question changes to "Now this is Flip 2. Which was better?"
- [ ] **4 intent buttons appear:**
  - Flip 1 was better (GAP Axis)
  - Flip 2 was better (RAM Axis)
  - Both Same
  - Repeat

### Phase F: JCC Power Right
- [ ] `jcc: "power_axis_switch"` sent on transition
- [ ] Auto-flip sequence restarts

### Expected History Log

```
14:00:00 - Test started
14:00:00 - Phoropter reset to 0/0/180   ← Only once!
14:00:01 - Chart: echart_400
14:02:00 - Chart: jcc_chart             ← Only once!
14:02:03 - Flip 2 displayed
14:02:10 - Response: Flip 1 was better (GAP Axis)
14:02:10 - JCC action: increase
14:02:10 - JCC action: handle
14:02:13 - Flip 2 displayed
14:02:20 - Response: Both Same
14:02:20 - JCC action: power_axis_switch
```

---

## Troubleshooting Tests

**Import errors:**
```bash
# Run from the tests directory
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests
python test_prev_state.py
```

**Tests fail unexpectedly:**
1. Verify backend is NOT running (tests use session directly)
2. Check all dependencies: `pip install flask flask-cors pyyaml`
3. Confirm `interactive_session.py` is in the parent directory

**Flip 2 intents missing (browser debugging):**
1. Check browser console for JavaScript errors
2. Network tab: verify AUTO_FLIP response contains `intents` array
3. Backend logs: confirm `is_flip2 = True` after `_update_state()`
4. Inspect DOM: check if buttons exist but are hidden

---

## Feature Coverage

| Feature | Tested |
|---------|--------|
| Prev State (Blurry) | ✅ |
| Prev State (Unable to read) | ✅ |
| Getting better removed | ✅ |
| JCC Power zero-cylinder | ✅ |
| BINO binocular balance | ✅ |
| Chart selector | ✅ |
| Duochrome reversal display | ✅ |
| Pinhole test | ✅ |

**Status:** All tests passing — run from `eye_test_engine/tests/`
