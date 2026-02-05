# JCC Phases: No set_power() Calls

## Confirmation

During JCC phases (E, F, H, I), we **DO NOT** use the "Set Eyes & Occluder (Combined)" API. We only use JCC-specific operations.

---

## JCC Phases

| Phase | Letter | Name | Operations Used |
|-------|--------|------|-----------------|
| `jcc_axis_right` | **E** | JCC Axis Right | `handle`, `increase`, `decrease` |
| `jcc_power_right` | **F** | JCC Power Right | `handle`, `increase`, `decrease`, `power_axis_switch` |
| `jcc_axis_left` | **H** | JCC Axis Left | `handle`, `increase`, `decrease` |
| `jcc_power_left` | **I** | JCC Power Left | `handle`, `increase`, `decrease`, `power_axis_switch` |

---

## What We DON'T Use During JCC Phases

### ❌ NO `set_power()` calls
```python
# This is NOT called during JCC phases
self.set_power(r_axis=self.current_row.r_axis, occluder="Left_Occluded")
self.set_power(r_cyl=self.current_row.r_cyl, occluder="Left_Occluded")
```

### ❌ NO "Set Eyes & Occluder (Combined)" API
```bash
# This API is NOT used during JCC phases
curl -X POST .../run-tests \
  -d '{"test_cases": [{"right_eye": {"axis": 95}, "aux_lens": "AuxLensL"}]}'
```

---

## What We DO Use During JCC Phases

### ✅ JCC Operations Only

#### 1. Handle (Flip between positions)
```python
self.jcc_flip("handle")
```
```bash
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "handle"}]}'
```

#### 2. Increase (Adjust value up)
```python
self.jcc_flip("increase")  # +5° for axis, +0.25D for power
```
```bash
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "increase"}]}'
```

#### 3. Decrease (Adjust value down)
```python
self.jcc_flip("decrease")  # -5° for axis, -0.25D for power
```
```bash
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "decrease"}]}'
```

#### 4. Power/Axis Switch (Mode transition)
```python
self.jcc_flip("power_axis_switch")  # Switch from Axis to Power mode
```
```bash
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "power_axis_switch"}]}'
```

---

## Implementation Verification

### Phase E: JCC Axis Right
```python
def _process_jcc_axis_right(self, intent: str) -> Dict:
    if "GAP Axis" in intent or "Flip 1" in intent:
        self.jcc_flip("increase")  # ✅ JCC operation only
        # Update internal state for tracking
        self.current_row.r_axis += 5
        self.jcc_flip("handle")
        # NO set_power() call ✅
```

### Phase F: JCC Power Right
```python
def _process_jcc_power_right(self, intent: str) -> Dict:
    if "GAP Power" in intent or "Flip 1" in intent:
        self.jcc_flip("increase")  # ✅ JCC operation only
        # Update internal state for tracking
        self.current_row.r_cyl += 0.25
        self.jcc_flip("handle")
        # NO set_power() call ✅
```

### Phase H: JCC Axis Left
```python
def _process_jcc_axis_left(self, intent: str) -> Dict:
    if "GAP Axis" in intent or "Flip 1" in intent:
        self.jcc_flip("increase")  # ✅ JCC operation only
        # Update internal state for tracking
        self.current_row.l_axis += 5
        self.jcc_flip("handle")
        # NO set_power() call ✅
```

### Phase I: JCC Power Left
```python
def _process_jcc_power_left(self, intent: str) -> Dict:
    if "GAP Power" in intent or "Flip 1" in intent:
        self.jcc_flip("increase")  # ✅ JCC operation only
        # Update internal state for tracking
        self.current_row.l_cyl += 0.25
        self.jcc_flip("handle")
        # NO set_power() call ✅
```

---

## Console Output During JCC Phases

### Phase E/F/H/I (JCC Phases)
```
✓ JCC action: increase
✓ JCC action: handle
```

**No power setting calls!** ✅

### Non-JCC Phases (A, B, D, G, J, K)
```
✓ Power set: R(-0.25/None/None) L(None/None/None) Occ: Left_Occluded
✓ JCC eye mode set: L
✓ Displaying: snellen_chart_200_150
```

**Power setting is used in non-JCC phases** ✅

---

## Why This Matters

1. **JCC Chart Controls Values**: The JCC chart and operations manage axis/power values internally
2. **No Conflicts**: Avoids conflicting commands to the phoropter
3. **Correct Workflow**: Follows the intended JCC operational flow
4. **Cleaner API**: Fewer redundant API calls
5. **Phoropter Manages State**: The phoropter handles its own JCC state

---

## Comparison: JCC vs Non-JCC Phases

| Aspect | JCC Phases (E,F,H,I) | Non-JCC Phases (A,B,D,G,J,K) |
|--------|----------------------|------------------------------|
| **Power Setting** | ❌ Not used | ✅ Used via `set_power()` |
| **Aux Lens** | ❌ Not set | ✅ Set (AuxLensL/R/OFF) |
| **JCC Operations** | ✅ Used (handle, increase, decrease) | ❌ Only eye mode (L/R/BINO) |
| **Value Changes** | ✅ Via JCC operations | ✅ Via `set_power()` |

---

## API Call Sequences

### JCC Phase (E: Axis Right)
```bash
# 1. Display JCC chart (once at phase start)
curl -X POST .../run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_19"]}}]}'

# 2. Flip to position 2
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "handle"}]}'

# 3. Patient selects - increase axis
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "increase"}]}'

# 4. Reset to position 1
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "handle"}]}'

# Repeat steps 2-4 until "Both Same"
```

### Non-JCC Phase (B: Right Eye Refraction)
```bash
# 1. Set power and occluder
curl -X POST .../run-tests \
  -d '{"test_cases": [{"right_eye": {"sph": -0.25}, "aux_lens": "AuxLensL"}]}'

# 2. Set JCC eye mode
curl -X POST .../run-tests \
  -d '{"test_cases": [{"jcc": "L"}]}'

# 3. Display chart
curl -X POST .../run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_10"]}}]}'
```

---

## Summary

✅ **Confirmed**: During JCC phases (E, F, H, I), we:
- **DO NOT** call `set_power()`
- **DO NOT** use "Set Eyes & Occluder (Combined)" API
- **ONLY** use JCC operations (`handle`, `increase`, `decrease`, `power_axis_switch`)

✅ **Verified**: The phoropter manages axis and cylinder values internally during JCC operations

✅ **Correct**: This follows the intended JCC workflow where the JCC chart controls its own state

---

## Date
February 5, 2026

## Status
✅ Verified - No `set_power()` calls during JCC phases (E, F, H, I)
