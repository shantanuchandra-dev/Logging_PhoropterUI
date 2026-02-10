# Chart Selector for Distance Vision (Phase A)

## Overview

The chart selector in Phase A (Distance Vision) now displays **all available charts** - both E-chart and Snellen charts. This gives the optometrist maximum flexibility to choose the most appropriate chart for establishing the patient's baseline vision.

## Feature Details

### Visual Display

The chart selector appears in **two phases**:
- **Phase A: Distance Vision** - Shows all available charts (E-chart + Snellen charts)
- **Phase B: Right/Left Eye Refraction** - Shows only Snellen charts (for automatic progression)

### Current Implementation

#### Phase A: Distance Vision
- **Available charts**: 8 total
  - 1 E-chart: `echart_400`
  - 7 Snellen charts: ranging from 20/200 to 20/10
- **Chart selector**: Displays all 8 options
- **Switching**: Full flexibility to choose any chart
- **Persistence**: Chart selection maintained through pinhole test

#### Phase B: Refraction
- **Available charts**: 7 Snellen charts only (20/200 to 20/10)
- **Chart selector**: Displays Snellen chart variations
- **Switching**: Active chart switching during examination
- **Auto-progression**: Automatic chart progression based on "Able to read" responses

## Technical Implementation

### Backend Changes

#### 1. Added All Charts List

In `interactive_session.py`:

```python
# All available charts for selection (Phase A)
self.all_charts = [
    "echart_400",
    "snellen_chart_200_150",
    "snellen_chart_100_80",
    "snellen_chart_70_60_50",
    "snellen_chart_40_30_25",
    "snellen_chart_25_20_15",
    "snellen_chart_20_20_20",
    "snellen_chart_20_15_10",
]

# Snellen charts only (Phase B - for automatic progression)
self.snellen_charts = [
    "snellen_chart_200_150",
    "snellen_chart_100_80",
    # ... etc
]
```

#### 2. Updated `start_distance_vision()`

Added `chart_info` with all charts to the initial response:

```python
def start_distance_vision(self):
    """Start Phase A: Distance Vision."""
    self.current_phase = "distance_vision"
    self.current_chart_index = 0
    self.set_chart(self.all_charts[0])  # Start with E-chart
    # ...
    
    return {
        "phase": self.phase_names[self.current_phase],
        "question": question,
        "intents": intents,
        "chart": self.all_charts[0],
        "occluder": "BINO",
        "power": { ... },
        "chart_info": {
            "available_charts": self.all_charts,  # All 8 charts
            "current_index": self.current_chart_index,
            "current_chart": self.all_charts[self.current_chart_index]
        }
    }
```

#### 3. Updated `_build_response()`

Extended to include chart_info with appropriate chart list based on phase:

```python
def _build_response(self) -> Dict:
    """Build response with current state."""
    # ... build base response ...
    
    # Add chart information for Phase A or Phase B
    if self.current_phase == "distance_vision":
        response["chart_info"] = {
            "available_charts": self.all_charts,  # All 8 charts
            "current_index": self.current_chart_index,
            "current_chart": self.all_charts[self.current_chart_index]
        }
    elif self.current_phase in ["right_eye_refraction", "left_eye_refraction"]:
        response["chart_info"] = {
            "available_charts": self.snellen_charts,  # Only Snellen
            "current_index": self.current_chart_index,
            "current_chart": self.snellen_charts[self.current_chart_index]
        }
    
    return response
```

#### 4. Updated `switch_chart()`

Extended to support distance vision phase with all charts:

```python
def switch_chart(self, chart_index: int) -> Dict:
    """Switch to a different chart during distance vision or refraction phase."""
    
    # Determine which chart list to use based on current phase
    if self.current_phase == "distance_vision":
        chart_list = self.all_charts  # All 8 charts
        phase_name = "distance vision"
    elif self.current_phase in ["right_eye_refraction", "left_eye_refraction"]:
        chart_list = self.snellen_charts  # Only Snellen
        phase_name = "refraction"
    else:
        raise ValueError(f"Chart switching not allowed in phase: {self.current_phase}")
    
    # Validate and switch chart
    # ...
```

### Frontend Changes

#### Updated `updateChartSelector()`

In `app.js`, extended the phase check to include Phase A:

```javascript
function updateChartSelector(data) {
    const chartSelector = document.getElementById('chartSelector');
    const chartGrid = document.getElementById('chartGrid');
    
    // Check if we're in Phase A (distance vision) or Phase B (refraction)
    const isPhaseA = data.phase && data.phase.includes('Distance Vision');
    const isPhaseB = data.phase && (
        data.phase.includes('Right Eye Refraction') || 
        data.phase.includes('Left Eye Refraction')
    );
    
    if ((isPhaseA || isPhaseB) && data.chart_info) {
        // Show chart selector
        chartSelector.classList.add('active');
        // ... build chart grid ...
    }
}
```

**No other frontend changes required!** The existing chart selector UI automatically adapts to show the available charts for each phase.

## User Experience

### Phase A: Distance Vision

```
┌─────────────────────────────────────────────┐
│ Phase A: Distance Vision (Step 2.1)        │
├─────────────────────────────────────────────┤
│                                             │
│ Q: Please read the line you can see clearly│
│                                             │
│ [1. Able to read]                           │
│ [2. Blurry]                                 │
│ [3. Unable to read]                         │
│                                             │
├─────────────────────────────────────────────┤
│ Charts:                                     │
│ ┌───────────────┐                           │
│ │ E-Chart       │ ← Currently selected      │
│ │ 20/400        │                           │
│ └───────────────┘                           │
└─────────────────────────────────────────────┘
```

### With Pinhole Test

The chart selector **persists** during the pinhole test:

```
┌─────────────────────────────────────────────┐
│ Phase A: Distance Vision (Step 2.1)        │
├─────────────────────────────────────────────┤
│                                             │
│ Q: With pinhole: Can you see the E clearly │
│    now?                                     │
│                                             │
│ [1. Able to read with pinhole]              │
│ [2. Still unable to read]                   │
│                                             │
├─────────────────────────────────────────────┤
│ Charts:                                     │
│ ┌───────────────┐                           │
│ │ E-Chart       │ ← Still available         │
│ │ 20/400        │                           │
│ └───────────────┘                           │
└─────────────────────────────────────────────┘
```

## Benefits

### 1. Maximum Flexibility
- Optometrist can choose **any chart** during distance vision baseline
- Can start with E-chart for patients with very poor vision
- Can switch to specific Snellen chart if patient can read better
- No restrictions during Phase A assessment

### 2. Clinical Utility
- **E-chart**: Best for patients with very poor baseline vision
- **Large Snellen (20/200-20/100)**: For patients with moderate impairment
- **Medium Snellen (20/70-20/40)**: For patients with mild impairment
- **Small Snellen (20/25-20/10)**: For patients with good baseline vision

### 3. Consistency
- Same chart switching mechanism across all phases
- Familiar UI pattern reduces learning curve

### 4. Workflow Optimization
- Quickly identify appropriate starting chart for refraction
- Save time by not progressing through unnecessary charts
- Immediate visual feedback on patient's baseline capability

## Future Enhancements

### Additional Chart Types

The system is designed to easily accommodate new chart types:

**To add new charts:**

1. **Add to chart mapping** in `interactive_session.py`:
```python
self.chart_map = {
    "echart_400": "chart_9",
    "echart_200": "chart_XX",  # New E-chart variation
    "duochrome_alt": "chart_YY",  # Alternative duochrome
    # ...
}
```

2. **Add to appropriate chart list**:
```python
# For Phase A (distance vision)
self.all_charts = [
    "echart_400",
    "echart_200",  # New chart automatically appears
    "snellen_chart_200_150",
    # ... rest of charts
]
```

3. **No other changes needed!** The UI will automatically display all charts.

### Potential New Charts

Future chart additions could include:
- **E-chart variations**: Different sizes (`echart_200`, `echart_100`)
- **Pediatric charts**: LEA symbols, pictures
- **Contrast charts**: Low contrast versions for special testing
- **LogMAR charts**: For research or specialized clinical settings
- **Letter charts**: C, D, or other letter-based charts

## Testing

### Test File: `test_distance_vision_chart_selector.py`

Located in: `eye_test_engine/tests/test_distance_vision_chart_selector.py`

#### Test Cases

1. **Chart Selector Present**
   - Verify `chart_info` included in distance vision response
   - Verify structure: `available_charts`, `current_index`, `current_chart`
   - Verify initial state is correct

2. **Chart Switching Works**
   - Test switching to different chart indices
   - Verify chart updates on phoropter
   - Verify state consistency

3. **Persistence Through Pinhole Test**
   - Start distance vision
   - Trigger pinhole test ("Unable to read")
   - Verify `chart_info` still present
   - Verify chart selector remains functional

#### Running the Tests

```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI
python3 eye_test_engine/tests/test_distance_vision_chart_selector.py
```

#### Expected Output

```
======================================================================
SUMMARY
======================================================================
Chart Selector Present:     ✅ PASSED
Chart Switching Works:      ✅ PASSED
Persistence Through Test:   ✅ PASSED

✅ ALL TESTS PASSED

Chart selector for distance vision verified:
✓ Chart info included in distance vision phase
✓ Chart switching works correctly
✓ Chart selector persists through pinhole test
```

## API Endpoints

The existing chart switching endpoint works for both phases:

```
POST /api/session/<session_id>/switch-chart
{
    "chart_index": 0
}
```

The backend automatically determines which chart list to use based on the current phase.

## Design Considerations

### Why Show Selector with One Chart?

Even though there's currently only one E-chart, we show the selector because:

1. **Visual feedback**: User sees which chart is active
2. **Future-proof**: Easy to add more charts later
3. **Consistency**: Same UI pattern reduces cognitive load
4. **No harm**: Takes minimal space, doesn't interfere with workflow

### Chart Selection During Pinhole

The chart selector remains active during pinhole test because:

1. **Optometrist may want to test different chart sizes** with pinhole
2. **State persistence** is simpler to maintain
3. **No clinical reason** to hide it

## Related Files

- **Backend Logic**: `eye_test_engine/interactive_session.py`
- **Frontend UI**: `eye_test_engine/frontend/app.js`
- **Test Suite**: `eye_test_engine/tests/test_distance_vision_chart_selector.py`
- **Previous Feature**: `CHART_SELECTOR_FEATURE.md` (Phase B implementation)

## See Also

- [Chart Selector Feature (Phase B)](./CHART_SELECTOR_FEATURE.md)
- [Pinhole Test Feature](./PINHOLE_TEST_FEATURE.md)
- [Distance Vision Phase Protocol](./eye_test_engine/config/protocol.yaml)
