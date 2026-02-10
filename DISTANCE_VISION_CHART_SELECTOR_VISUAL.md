# Distance Vision Chart Selector - Visual Guide

## UI Layout

### Phase A: Distance Vision - Default State

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 👁️ Eye Test Engine                          [Set AR] [Set Lenso]┃
┃ Interactive Phoropter-Controlled Eye Examination                ┃
┃                                            [Apply: AR | Lenso]  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┌─────────────────────────────────────────────────────────────────┐
│ Phase Badge: Phase A: Distance Vision (Step 2.1)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Current Power Display                                           │
│ ┌────────────────┐                      ┌────────────────┐      │
│ │ Right Eye      │                      │ Left Eye       │      │
│ │ SPH: 0.00      │                      │ SPH: 0.00      │      │
│ │ CYL: 0.00      │                      │ CYL: 0.00      │      │
│ │ AXIS: 180°     │                      │ AXIS: 180°     │      │
│ └────────────────┘                      └────────────────┘      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Question:                                                       │
│                                                                 │
│ Please read the line you can see clearly.                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Intents:                                                        │
│                                                                 │
│  ┌──────────────────┐ ┌──────────────────┐ ┌─────────────────┐│
│  │ 1. Able to read  │ │ 2. Blurry        │ │ 3. Unable to    ││
│  │                  │ │                  │ │    read         ││
│  └──────────────────┘ └──────────────────┘ └─────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Chart Selection:                                                │
│                                                                 │
│  ┏━━━━━━━━━━━━━━━┓                                              │
│  ┃ E-Chart       ┃ ← Active (blue border)                      │
│  ┃ 20/400        ┃                                              │
│  ┗━━━━━━━━━━━━━━━┛                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Footer: Jump to Phase                                          ┃
┃ [Phase A] [Phase B] [Phase C] ...                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Phase A: After "Unable to Read" - Pinhole Test

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 👁️ Eye Test Engine                          [Set AR] [Set Lenso]┃
┃ Interactive Phoropter-Controlled Eye Examination                ┃
┃                                            [Apply: AR | Lenso]  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

┌─────────────────────────────────────────────────────────────────┐
│ Phase Badge: Phase A: Distance Vision (Step 2.1)               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Question:                                                       │
│                                                                 │
│ 📌 With pinhole: Can you see the E clearly now?                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Intents:                                                        │
│                                                                 │
│  ┌────────────────────────────┐  ┌──────────────────────────┐  │
│  │ 1. Able to read with       │  │ 2. Still unable to read  │  │
│  │    pinhole                 │  │                          │  │
│  └────────────────────────────┘  └──────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Chart Selection:                     ← STILL VISIBLE            │
│                                                                 │
│  ┏━━━━━━━━━━━━━━━┓                                              │
│  ┃ E-Chart       ┃ ← Still active during pinhole                │
│  ┃ 20/400        ┃                                              │
│  ┗━━━━━━━━━━━━━━━┛                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Comparison: Phase A vs Phase B

### Phase A: Distance Vision (Single Chart)

```
┌───────────────────────────────┐
│ Chart Selection:              │
│                               │
│  ┏━━━━━━━━━━━━━━━┓            │
│  ┃ E-Chart       ┃            │
│  ┃ 20/400        ┃            │
│  ┗━━━━━━━━━━━━━━━┛            │
│                               │
└───────────────────────────────┘
```

### Phase B: Refraction (Multiple Charts)

```
┌───────────────────────────────────────────────────────────┐
│ Chart Selection:                                          │
│                                                           │
│  ┏━━━━━━━━━━━━━┓ ┌─────────────┐ ┌─────────────┐        │
│  ┃ Chart       ┃ │ Chart       │ │ Chart       │        │
│  ┃ 20/200-20/150┃ │ 20/100-20/80│ │ 20/70-20/60 │        │
│  ┗━━━━━━━━━━━━━┛ └─────────────┘ └─────────────┘        │
│                                                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │ Chart       │ │ Chart       │ │ Chart       │        │
│  │ 20/40-20/30 │ │ 20/25-20/20 │ │ 20/20-20/20 │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
│                                                           │
│  ┌─────────────┐                                         │
│  │ Chart       │                                         │
│  │ 20/20-20/15 │                                         │
│  └─────────────┘                                         │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

## Future State with Multiple E-Charts

If more E-charts are added (e.g., `echart_200`, `echart_100`):

```
┌─────────────────────────────────────────────────────────────────┐
│ Chart Selection:                                                │
│                                                                 │
│  ┏━━━━━━━━━━━━━━━┓ ┌───────────────┐ ┌───────────────┐        │
│  ┃ E-Chart       ┃ │ E-Chart       │ │ E-Chart       │        │
│  ┃ 20/400        ┃ │ 20/200        │ │ 20/100        │        │
│  ┗━━━━━━━━━━━━━━━┛ └───────────────┘ └───────────────┘        │
│   ↑ Active                                                      │
│                                                                 │
│  ┌───────────────┐ ┌───────────────┐                           │
│  │ E-Chart       │ │ E-Chart       │                           │
│  │ High Contrast │ │ Pediatric     │                           │
│  └───────────────┘ └───────────────┘                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Visual States

### Active Chart Button

```
┏━━━━━━━━━━━━━━━━━━┓
┃ E-Chart          ┃ ← Blue border (#007bff)
┃ 20/400           ┃    Blue background (#e7f3ff)
┗━━━━━━━━━━━━━━━━━━┛    Bold text
```

### Inactive Chart Button

```
┌──────────────────┐
│ E-Chart          │ ← Gray border (#ddd)
│ 20/400           │    White background
└──────────────────┘    Normal text
```

### Hover State

```
┌──────────────────┐
│ E-Chart          │ ← Lighter border
│ 20/400           │    Light gray background (#f8f9fa)
└──────────────────┘    Slightly elevated (shadow)
```

## Responsive Design

### Desktop (> 768px)

Charts displayed in grid with 3 columns:

```
┌────┐ ┌────┐ ┌────┐
│ C1 │ │ C2 │ │ C3 │
└────┘ └────┘ └────┘
```

### Tablet/Mobile (< 768px)

Charts stack vertically:

```
┌────────┐
│   C1   │
└────────┘
┌────────┐
│   C2   │
└────────┘
┌────────┐
│   C3   │
└────────┘
```

## CSS Styling Reference

Key CSS classes used:

- `.chart-selector` - Container for the chart selector section
- `.chart-grid` - Grid layout for chart buttons
- `.chart-button` - Individual chart button
- `.chart-button.active` - Active/selected chart (blue styling)
- `.chart-name` - Chart name text
- `.chart-size` - Visual acuity range text
- `.chart-info` - Additional chart information

## Accessibility

- **Keyboard Navigation**: Chart buttons are focusable with Tab key
- **Click Target**: Large buttons (150px × 120px minimum) for easy clicking
- **Visual Feedback**: Clear active state with color and border
- **Screen Readers**: Buttons include descriptive text for chart names

## Animation

- **Fade In**: Chart selector fades in when entering Phase A/B
- **Smooth Transition**: Active state changes with smooth border/background transition
- **Hover Effect**: Subtle elevation and shadow on hover

## Technical Notes

- Chart selector auto-hides in JCC and Duochrome phases
- Chart switching sends CURL command to phoropter immediately
- UI updates automatically after successful chart switch
- No page reload required - all updates via AJAX

---

**Related Documentation:**
- [Distance Vision Chart Selector](./DISTANCE_VISION_CHART_SELECTOR.md)
- [Chart Selector Feature (Phase B)](./CHART_SELECTOR_FEATURE.md)
- [Chart Selector Visual Guide (Phase B)](./CHART_SELECTOR_VISUAL_GUIDE.md)
