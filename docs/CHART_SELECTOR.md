# Chart Selector Feature

## Overview

The Chart Selector allows switching between available charts during Phase A (Distance Vision) and Phase B (Right/Left Eye Refraction) without disrupting the test state. It provides visual flexibility so the examiner can jump to any chart without following a linear progression.

---

## Available Charts by Phase

### Phase A: Distance Vision — 8 charts

| # | Chart Name | Visual Acuity |
|---|-----------|---------------|
| 1 | `echart_400` | 20/400 (E-Chart) |
| 2 | `snellen_chart_200_150` | 20/200 – 20/150 |
| 3 | `snellen_chart_100_80` | 20/100 – 20/80 |
| 4 | `snellen_chart_70_60_50` | 20/70 – 20/50 |
| 5 | `snellen_chart_40_30_25` | 20/40 – 20/25 |
| 6 | `snellen_chart_25_20_15` | 20/25 – 20/15 |
| 7 | `snellen_chart_20_20_20` | 20/20 (target) |
| 8 | `snellen_chart_20_15_10` | 20/15 – 20/10 |

### Phase B: Right/Left Eye Refraction — 7 Snellen charts only

Charts 2–8 from the table above (E-chart excluded).

---

## UI Layout

### Position

Chart selector appears **below the intent buttons**:

```
┌─────────────────────────────────────────────┐
│ Question: I'm covering your left eye...     │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ [1. Able to read] [2. Blurry] [3. Unable]  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│ 📊 Chart Selection                          │
│ ℹ️  Click any chart to switch.              │
│                                             │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│ │Chart     │ │Chart     │ │Chart     │    │
│ │200/150   │ │100/80    │ │70/60/50  │    │
│ └──────────┘ └──────────┘ └──────────┘    │
│                                             │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│ │Chart     │ │Chart     │ │20/20/20  │    │
│ │40/30/25  │ │25/20/15  │ │(ACTIVE)  │    │
│ └──────────┘ └──────────┘ └──────────┘    │
└─────────────────────────────────────────────┘
```

### Button States

| State | Appearance |
|-------|-----------|
| Inactive | Gray background (#f5f5f5), black text |
| Hover | Light blue (#e3f2fd), purple border, slight lift |
| Active | Purple-blue gradient, white text, bold |

### Visible During

- Phase A: Distance Vision
- Phase B: Right/Left Eye Refraction
- Persists during Pinhole test in Phase A

### Hidden During

- JCC Axis/Power phases
- Duochrome phases
- Binocular Balance

---

## Usage Examples

| Scenario | Action |
|----------|--------|
| Patient reads easily on 200/150 | Click "Chart 25/20/15" to skip ahead |
| Patient can't read 40/30/25 | Click "Chart 70/60/50" to go back |
| Phase A — patient seems too good for E-chart | Click any Snellen chart directly |

---

## Technical Implementation

### Backend Changes (`interactive_session.py`)

**Chart Lists:**
```python
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

self.snellen_charts = self.all_charts[1:]  # All except E-chart
```

**`switch_chart()` Method:**
```python
def switch_chart(self, chart_index: int) -> Dict:
    if self.current_phase == "distance_vision":
        chart_list = self.all_charts
    elif self.current_phase in ["right_eye_refraction", "left_eye_refraction"]:
        chart_list = self.snellen_charts
    else:
        raise ValueError(f"Chart switching not allowed in phase: {self.current_phase}")

    self.current_chart_index = chart_index
    chart_name = chart_list[chart_index]
    self.set_chart(chart_name)
    self.current_row.chart_display = chart_name
    return self._build_response()
```

**`_build_response()` — chart_info added:**
```python
if self.current_phase == "distance_vision":
    response["chart_info"] = {
        "available_charts": self.all_charts,
        "current_index": self.current_chart_index,
        "current_chart": self.all_charts[self.current_chart_index]
    }
elif self.current_phase in ["right_eye_refraction", "left_eye_refraction"]:
    response["chart_info"] = {
        "available_charts": self.snellen_charts,
        "current_index": self.current_chart_index,
        "current_chart": self.snellen_charts[self.current_chart_index]
    }
```

### API Endpoint (`api_server.py`)

**`POST /api/session/<id>/switch-chart`**

Request:
```json
{ "chart_index": 3 }
```

Success Response (200):
```json
{
  "chart": "snellen_chart_40_30_25",
  "chart_info": {
    "available_charts": ["snellen_chart_200_150", "..."],
    "current_index": 3,
    "current_chart": "snellen_chart_40_30_25"
  },
  "power": { "right": {...}, "left": {...} }
}
```

Error Responses: `400` (invalid index/wrong phase), `404` (session not found)

### Frontend Changes

**`index.html`** — Added `#chartSelector` div with `.chart-grid` below intents section.

**`app.js`** — New functions:
- `updateChartSelector(data)` — Show/hide and populate grid based on phase
- `formatChartName(chartName)` — "snellen_chart_200_150" → "Chart 200/150"
- `switchChart(chartIndex)` — POST to switch-chart endpoint
- Updates `sessionState.availableCharts` and `sessionState.currentChartIndex`

**Interaction flow:**
1. User clicks a chart button
2. Frontend calls `POST /api/session/{id}/switch-chart`
3. Backend validates, calls `set_chart()`, returns updated state
4. Frontend updates active chart highlighting, power display unchanged
5. History log: "Switched to chart 6"

---

## Chart VA Size Selection

For charts that have multiple rows, you can select which size line to highlight:

| Chart ID | Size Options |
|----------|-------------|
| chart_10 | 200, 150 |
| chart_11 | 100, 80 |
| chart_12 | 70, 60, 50 |
| chart_13 | 40, 30, 25 |
| chart_14 | 20, 15, 10 |
| chart_15 | 20_1, 20_2, 20_3 |
| chart_16 | 25, 20, 15 |
| chart_20 | R, L |

```bash
# Example: Snellen 100/80, highlight size 100
curl -X POST .../phoropter/{PHOROPTER_ID}/run-tests \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_11", "100"]}}]}'
```

---

## Testing

### Automated Test

`eye_test_engine/tests/test_distance_vision_chart_selector.py`

```bash
python3 eye_test_engine/tests/test_distance_vision_chart_selector.py
```

Expected:
```
Chart Selector Present:     ✅ PASSED
Chart Switching Works:      ✅ PASSED
Persistence Through Test:   ✅ PASSED
```

### Manual Test Steps

1. Start test, reach Phase A
2. Verify chart selector appears with E-chart highlighted
3. Click "Chart 25/20/15" — verify phoropter switches, power unchanged
4. Reach Phase B (Right Eye Refraction)
5. Verify only Snellen charts shown (no E-chart)
6. Click different charts, verify smooth transitions
7. Progress to JCC — verify chart selector disappears
8. Return to left eye refraction — verify it reappears

---

## Future Enhancements

- Keyboard shortcuts (1–7) to switch charts
- History tracking per chart
- LogMAR / pediatric chart additions
- Chart recommendations based on patient responses

---

## Files Modified

- `eye_test_engine/interactive_session.py` — `switch_chart()`, `_build_response()`, `start_distance_vision()`
- `eye_test_engine/api_server.py` — `/api/session/<id>/switch-chart` endpoint
- `eye_test_engine/frontend/index.html` — Chart selector HTML + CSS
- `eye_test_engine/frontend/app.js` — Chart selector JavaScript

**Status:** ✅ Complete — Fully backward compatible
