# JCC Adjustment Step Sizes

## Configuration

The JCC adjustment step sizes are correctly configured and implemented.

### Axis Refinement
**Step Size: ±5 degrees**

### Cylinder Power Refinement
**Step Size: ±0.25 diopters**

---

## Configuration File

Located in `config/thresholds.yaml`:

```yaml
cylinder_refinement:
  axis_increment: 5      # Degrees to adjust per flip
  power_increment: 0.25  # Diopters to adjust per flip
```

---

## Implementation

### Axis Adjustments (Right Eye)

**Increase Axis (+5°):**
```python
# When patient selects Flip 1 (GAP Axis)
self.current_row.r_axis += 5
if self.current_row.r_axis > 180:
    self.current_row.r_axis -= 180
self.set_power(r_axis=self.current_row.r_axis, occluder="Left_Occluded")
self.jcc_flip("increase")
```

**Decrease Axis (-5°):**
```python
# When patient selects Flip 2 (RAM Axis)
self.current_row.r_axis -= 5
if self.current_row.r_axis < 0:
    self.current_row.r_axis += 180
self.set_power(r_axis=self.current_row.r_axis, occluder="Left_Occluded")
self.jcc_flip("decrease")
```

### Axis Adjustments (Left Eye)

**Increase Axis (+5°):**
```python
# When patient selects Flip 1 (GAP Axis)
self.current_row.l_axis += 5
if self.current_row.l_axis > 180:
    self.current_row.l_axis -= 180
self.set_power(l_axis=self.current_row.l_axis, occluder="Right_Occluded")
self.jcc_flip("increase")
```

**Decrease Axis (-5°):**
```python
# When patient selects Flip 2 (RAM Axis)
self.current_row.l_axis -= 5
if self.current_row.l_axis < 0:
    self.current_row.l_axis += 180
self.set_power(l_axis=self.current_row.l_axis, occluder="Right_Occluded")
self.jcc_flip("decrease")
```

### Cylinder Power Adjustments (Right Eye)

**Increase Cylinder (+0.25D):**
```python
# When patient selects Flip 1 (GAP Power)
self.current_row.r_cyl += 0.25
self.set_power(r_cyl=self.current_row.r_cyl, occluder="Left_Occluded")
self.jcc_flip("increase")
```

**Decrease Cylinder (-0.25D):**
```python
# When patient selects Flip 2 (RAM Power)
self.current_row.r_cyl -= 0.25
self.set_power(r_cyl=self.current_row.r_cyl, occluder="Left_Occluded")
self.jcc_flip("decrease")
```

### Cylinder Power Adjustments (Left Eye)

**Increase Cylinder (+0.25D):**
```python
# When patient selects Flip 1 (GAP Power)
self.current_row.l_cyl += 0.25
self.set_power(l_cyl=self.current_row.l_cyl, occluder="Right_Occluded")
self.jcc_flip("increase")
```

**Decrease Cylinder (-0.25D):**
```python
# When patient selects Flip 2 (RAM Power)
self.current_row.l_cyl -= 0.25
self.set_power(l_cyl=self.current_row.l_cyl, occluder="Right_Occluded")
self.jcc_flip("decrease")
```

---

## Axis Wraparound Logic

### Forward Wraparound (> 180°)
When axis exceeds 180°, it wraps back to the beginning:
```python
if self.current_row.r_axis > 180:
    self.current_row.r_axis -= 180
```

**Example:**
- Current: 178°
- Increase: +5° = 183°
- Wrapped: 183° - 180° = 3°

### Backward Wraparound (< 0°)
When axis goes below 0°, it wraps to the end:
```python
if self.current_row.r_axis < 0:
    self.current_row.r_axis += 180
```

**Example:**
- Current: 2°
- Decrease: -5° = -3°
- Wrapped: -3° + 180° = 177°

---

## Code Locations

### Interactive Session (`interactive_session.py`)

| Method | Lines | Adjustment |
|--------|-------|------------|
| `_process_jcc_axis_right()` | 409, 427 | Right axis ±5° |
| `_process_jcc_axis_left()` | 479, 494 | Left axis ±5° |
| `_process_jcc_power_right()` | 542, 556 | Right cyl ±0.25D |
| `_process_jcc_power_left()` | 602, 615 | Left cyl ±0.25D |

---

## Verification

### Test Axis Adjustments
1. Start JCC Axis Right phase
2. Initial axis: 90°
3. Select "Flip 1 was better" → Axis becomes 95° (+5°)
4. Select "Flip 1 was better" → Axis becomes 100° (+5°)
5. Select "Flip 2 was better" → Axis becomes 95° (-5°)

### Test Power Adjustments
1. Start JCC Power Right phase
2. Initial cylinder: -1.00D
3. Select "Flip 1 was better" → Cylinder becomes -0.75D (+0.25D)
4. Select "Flip 1 was better" → Cylinder becomes -0.50D (+0.25D)
5. Select "Flip 2 was better" → Cylinder becomes -0.75D (-0.25D)

### Test Axis Wraparound
1. Set axis to 178°
2. Select "Flip 1 was better" → Axis becomes 3° (wrapped)
3. Set axis to 2°
4. Select "Flip 2 was better" → Axis becomes 177° (wrapped)

---

## Summary

✅ **Axis Step Size**: 5 degrees (correctly implemented)
✅ **Cylinder Step Size**: 0.25 diopters (correctly implemented)
✅ **Wraparound Logic**: Properly handles 0°-180° range
✅ **Configuration**: Defined in `thresholds.yaml`
✅ **All Four Phases**: Right/Left Axis and Right/Left Power

## Date
February 5, 2026

## Status
✅ Verified - JCC adjustment step sizes are correctly implemented
