# Chart Selector Feature for Phase B

## Overview

During Phase B (Right Eye Refraction and Left Eye Refraction), the UI now displays all available Snellen charts with the ability to switch between them at any time. This allows the examiner to jump to any chart and proceed from there without following the linear progression.

## Features

### Visual Chart Grid
- **Grid Layout**: All 7 available charts displayed in a responsive grid
- **Current Chart Highlighting**: Active chart is highlighted with gradient background
- **Chart Information**: Each chart shows:
  - Chart name (e.g., "Chart 200/150")
  - Visual acuity range (e.g., "20/200 - 20/150")

### Available Charts

1. **Chart 200/150** - 20/200 - 20/150
2. **Chart 100/80** - 20/100 - 20/80
3. **Chart 70/60/50** - 20/70 - 20/60 - 20/50
4. **Chart 40/30/25** - 20/40 - 20/30 - 20/25
5. **Chart 25/20/15** - 20/25 - 20/20 - 20/15
6. **Chart 20/20/20** - 20/20 (target)
7. **Chart 20/15/10** - 20/20 - 20/15 - 20/10

### Functionality

- **Click to Switch**: Click any chart button to immediately switch to that chart
- **Maintains State**: Current power settings are preserved when switching charts
- **Real-time Update**: Phoropter chart updates immediately via CURL command
- **Progress Tracking**: Chart index is tracked and can be used to resume testing

## User Interface

### When Active (Phase B Only)

The chart selector appears **below the question and intent buttons**:

```
┌─────────────────────────────────────────────┐
│ Question: I'm covering your left eye...     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ Please select your response:                │
│ [1. Able to read] [2. Blurry] [3. Unable]  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ 📊 Chart Selection                          │
│ ℹ️  Click any chart below to switch to it. │
│     Current progress will be maintained.    │
│                                             │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│ │Chart     │ │Chart     │ │Chart     │    │
│ │200/150   │ │100/80    │ │70/60/50  │    │
│ │20/200-   │ │20/100-   │ │20/70-    │    │
│ │20/150    │ │20/80     │ │20/60-20/50│   │
│ └──────────┘ └──────────┘ └──────────┘    │
│                                             │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│ │Chart     │ │Chart     │ │Chart     │    │
│ │40/30/25  │ │25/20/15  │ │20/20/20  │    │
│ │20/40-    │ │20/25-    │ │20/20     │    │
│ │20/30-    │ │20/20-    │ │(Active)  │    │
│ │20/25     │ │20/15     │ │          │    │
│ └──────────┘ └──────────┘ └──────────┘    │
│                                             │
│ ┌──────────┐                                │
│ │Chart     │                                │
│ │20/15/10  │                                │
│ │20/20-    │                                │
│ │20/15-    │                                │
│ │20/10     │                                │
│ └──────────┘                                │
└─────────────────────────────────────────────┘
```

### When Inactive (Other Phases)
The chart selector is hidden during:
- Phase A: Distance Vision
- JCC Axis refinement
- JCC Power refinement
- Duochrome tests
- Binocular balance

## Technical Implementation

### Backend Changes

#### 1. `interactive_session.py`

**Added `switch_chart()` method**:
```python
def switch_chart(self, chart_index: int) -> Dict:
    """Switch to a different chart during refraction phase."""
    # Validates phase and chart index
    # Updates current_chart_index
    # Sets chart on phoropter
    # Returns updated state
```

**Updated `_build_response()` method**:
```python
# Add chart information if in Phase B
if self.current_phase in ["right_eye_refraction", "left_eye_refraction"]:
    response["chart_info"] = {
        "available_charts": self.snellen_charts,
        "current_index": self.current_chart_index,
        "current_chart": self.snellen_charts[self.current_chart_index]
    }
```

#### 2. `api_server.py`

**New endpoint**: `POST /api/session/<id>/switch-chart`
```json
Request:
{
  "chart_index": 3
}

Response:
{
  "session_id": "session_123",
  "status": "active",
  "phase": "Phase B: Right Eye Refraction",
  "chart_info": {
    "available_charts": [...],
    "current_index": 3,
    "current_chart": "snellen_chart_40_30_25"
  },
  ...
}
```

### Frontend Changes

#### 1. `index.html`

**Added CSS styles**:
- `.chart-selector` - Container for chart selection UI
- `.chart-grid` - Responsive grid layout for chart buttons
- `.chart-button` - Individual chart button styling
- `.chart-button.active` - Highlighted active chart

**Added HTML structure** (positioned after intents):
```html
<!-- Question box -->
<div class="question-box">...</div>

<!-- Intents (response buttons) -->
<div class="intents-container">...</div>

<!-- Chart selector (below intents) -->
<div id="chartSelector" class="chart-selector">
    <h4>Chart Selection</h4>
    <div class="chart-info">...</div>
    <div id="chartGrid" class="chart-grid"></div>
</div>
```

#### 2. `app.js`

**New functions**:
- `updateChartSelector(data)` - Show/hide and populate chart grid
- `formatChartName(chartName)` - Convert chart name to display format
- `extractChartSize(chartName)` - Extract visual acuity range
- `switchChart(chartIndex)` - Handle chart switching

**Updated functions**:
- `displayQuestion(data)` - Now calls `updateChartSelector()`

**Session state tracking**:
```javascript
sessionState.availableCharts = [];
sessionState.currentChartIndex = 0;
```

## Usage Examples

### Example 1: Jump to Target Chart
1. Start test, reach Phase B
2. Currently on Chart 200/150
3. Click "Chart 20/20/20" button
4. System immediately switches to 20/20 chart
5. Continue testing from there

### Example 2: Go Back to Larger Chart
1. Patient says "Unable to read" on Chart 40/30/25
2. Instead of clicking "Unable to read" again
3. Click "Chart 70/60/50" to go back to a larger chart
4. Continue testing

### Example 3: Skip Intermediate Charts
1. Patient reads Chart 200/150 easily
2. Instead of going through each chart sequentially
3. Jump directly to Chart 25/20/15
4. Save time by skipping intermediate steps

## Benefits

1. **Flexibility**: Examiner can adapt to patient's visual acuity
2. **Time Saving**: Skip charts that are clearly too easy or too hard
3. **Better UX**: Visual grid makes it easy to see all options
4. **Non-disruptive**: Maintains all power settings and test state
5. **Professional**: Clean, modern UI that matches the rest of the interface

## API Reference

### Switch Chart Endpoint

**Endpoint**: `POST /api/session/<session_id>/switch-chart`

**Request Body**:
```json
{
  "chart_index": 3
}
```

**Success Response** (200):
```json
{
  "session_id": "session_123",
  "status": "active",
  "phase": "Phase B: Right Eye Refraction (Step 6.1)",
  "question": "I'm covering your left eye...",
  "intents": ["Able to read", "Blurry", "Unable to read"],
  "chart": "snellen_chart_40_30_25",
  "chart_info": {
    "available_charts": [
      "snellen_chart_200_150",
      "snellen_chart_100_80",
      "snellen_chart_70_60_50",
      "snellen_chart_40_30_25",
      "snellen_chart_25_20_15",
      "snellen_chart_20_20_20",
      "snellen_chart_20_15_10"
    ],
    "current_index": 3,
    "current_chart": "snellen_chart_40_30_25"
  },
  "occluder": "Left_Occluded",
  "power": {
    "right": {"sph": -1.25, "cyl": -0.50, "axis": 180},
    "left": {"sph": 0.0, "cyl": 0.0, "axis": 180}
  }
}
```

**Error Responses**:
- `404`: Session not found
- `400`: Invalid chart index or not in refraction phase
- `500`: Server error

## Testing

### Manual Testing Steps

1. **Start Test**:
   ```bash
   cd eye_test_engine
   python3 api_server.py
   ```

2. **Open Frontend**:
   - Open `eye_test_engine/frontend/index.html` in browser
   - Click "Start Test"

3. **Navigate to Phase B**:
   - Click "Able to read" for distance vision
   - You should now be in "Phase B: Right Eye Refraction"

4. **Verify Chart Selector Appears**:
   - Chart selector should be visible below the question
   - All 7 charts should be displayed in grid
   - First chart should be highlighted (active)

5. **Test Chart Switching**:
   - Click on "Chart 20/20/20" (6th chart)
   - Verify:
     - Chart 20/20/20 becomes highlighted
     - CURL command is sent to phoropter
     - Power display remains unchanged
     - Question and intents remain the same

6. **Test Multiple Switches**:
   - Click different charts multiple times
   - Verify smooth transitions
   - Check browser console for errors

7. **Test Phase Transitions**:
   - Progress to JCC Axis phase
   - Verify chart selector disappears
   - Return to left eye refraction
   - Verify chart selector reappears

### Automated Testing

Create test file: `eye_test_engine/tests/test_chart_switching.py`

```python
def test_chart_switching():
    session = InteractiveSession()
    session.start_distance_vision()
    session.process_response("Able to read")
    
    # Should be in right eye refraction
    assert session.current_phase == "right_eye_refraction"
    assert session.current_chart_index == 0
    
    # Switch to chart 3
    response = session.switch_chart(3)
    assert response["chart_info"]["current_index"] == 3
    assert session.current_chart_index == 3
    
    # Verify chart was set
    assert response["chart"] == "snellen_chart_40_30_25"
```

## Files Modified

1. `eye_test_engine/interactive_session.py`
   - Added `switch_chart()` method
   - Updated `_build_response()` to include chart_info

2. `eye_test_engine/api_server.py`
   - Added `/api/session/<id>/switch-chart` endpoint

3. `eye_test_engine/frontend/index.html`
   - Added chart selector CSS styles
   - Added chart selector HTML structure

4. `eye_test_engine/frontend/app.js`
   - Added chart selector JavaScript functions
   - Updated session state tracking
   - Updated displayQuestion() to show/hide selector

## Files Created

1. `CHART_SELECTOR_FEATURE.md` - This documentation file

## Future Enhancements

1. **Keyboard Shortcuts**: Number keys 1-7 to switch charts
2. **Chart Preview**: Show chart content preview on hover
3. **History Tracking**: Show which charts have been tested
4. **Quick Jump**: "Jump to 20/20" button for common use case
5. **Chart Recommendations**: Suggest next chart based on responses
6. **Mobile Optimization**: Better touch targets for tablet use
