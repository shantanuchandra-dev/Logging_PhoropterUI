# Chart Selector Visual Guide

## Overview

This guide shows what the chart selector looks like in the UI during Phase B (Right Eye and Left Eye Refraction).

## Full UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    Eye Test Engine                              │
│                Interactive Session                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase: Phase B: Right Eye Refraction (Step 6.1)              │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ Question                                                   │ │
│  │ I'm covering your left eye. Please read the line you can  │ │
│  │ see clearly.                                               │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Please select your response:                                  │
│  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐   │
│  │ 1. Able to     │ │ 2. Blurry      │ │ 3. Unable to   │   │
│  │    read        │ │                │ │    read        │   │
│  └────────────────┘ └────────────────┘ └────────────────┘   │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ 📊 Chart Selection                                        │ │
│  │ ───────────────────────────────────────────────────────── │ │
│  │ ℹ️  Click any chart below to switch to it. Current       │ │
│  │     progress will be maintained.                          │ │
│  │                                                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │ │
│  │  │ Chart       │  │ Chart       │  │ Chart       │      │ │
│  │  │ 200/150     │  │ 100/80      │  │ 70/60/50    │      │ │
│  │  │             │  │             │  │             │      │ │
│  │  │ 20/200 -    │  │ 20/100 -    │  │ 20/70 -     │      │ │
│  │  │ 20/150      │  │ 20/80       │  │ 20/60 -     │      │ │
│  │  │             │  │             │  │ 20/50       │      │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │ │
│  │                                                            │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │ │
│  │  │ Chart       │  │ Chart       │  │ Chart       │      │ │
│  │  │ 40/30/25    │  │ 25/20/15    │  │ 20/20/20    │      │ │
│  │  │             │  │             │  │             │      │ │
│  │  │ 20/40 -     │  │ 20/25 -     │  │ 20/20       │      │ │
│  │  │ 20/30 -     │  │ 20/20 -     │  │ (ACTIVE)    │      │ │
│  │  │ 20/25       │  │ 20/15       │  │             │      │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │ │
│  │                                        ↑ Highlighted      │ │
│  │  ┌─────────────┐                                          │ │
│  │  │ Chart       │                                          │ │
│  │  │ 20/15/10    │                                          │ │
│  │  │             │                                          │ │
│  │  │ 20/20 -     │                                          │ │
│  │  │ 20/15 -     │                                          │ │
│  │  │ 20/10       │                                          │ │
│  │  └─────────────┘                                          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Power: R: -1.25 / -0.50 / 180°    L: 0.00 / 0.00 / 180°    │
│  Occluder: Left_Occluded                                      │
└─────────────────────────────────────────────────────────────────┘
```

## Chart Button States

### Inactive Chart (Default State)
```
┌─────────────┐
│ Chart       │  ← Gray background (#f5f5f5)
│ 200/150     │  ← Black text
│             │
│ 20/200 -    │  ← Slightly faded text
│ 20/150      │
│             │
└─────────────┘
```

### Hover State
```
┌─────────────┐
│ Chart       │  ← Light blue background (#e3f2fd)
│ 200/150     │  ← Slightly elevated (translateY(-2px))
│             │  ← Blue border (#667eea)
│ 20/200 -    │  ← Cursor: pointer
│ 20/150      │
│             │
└─────────────┘
```

### Active Chart (Current)
```
┌─────────────┐
│ Chart       │  ← Gradient background (purple-blue)
│ 20/20/20    │  ← White text, bold
│             │  ← Blue border (#667eea)
│ 20/20       │  ← White text
│ (ACTIVE)    │  ← Clear indicator
│             │
└─────────────┘
```

## Responsive Behavior

### Desktop (Wide Screen)
```
Grid: 3 columns × 3 rows (7 charts + 2 empty cells)

┌────────┐ ┌────────┐ ┌────────┐
│Chart 1 │ │Chart 2 │ │Chart 3 │
└────────┘ └────────┘ └────────┘

┌────────┐ ┌────────┐ ┌────────┐
│Chart 4 │ │Chart 5 │ │Chart 6 │
└────────┘ └────────┘ └────────┘

┌────────┐
│Chart 7 │
└────────┘
```

### Tablet (Medium Screen)
```
Grid: 2 columns × 4 rows

┌────────┐ ┌────────┐
│Chart 1 │ │Chart 2 │
└────────┘ └────────┘

┌────────┐ ┌────────┐
│Chart 3 │ │Chart 4 │
└────────┘ └────────┘

┌────────┐ ┌────────┐
│Chart 5 │ │Chart 6 │
└────────┘ └────────┘

┌────────┐
│Chart 7 │
└────────┘
```

### Mobile (Narrow Screen)
```
Grid: 1 column × 7 rows

┌────────┐
│Chart 1 │
└────────┘
┌────────┐
│Chart 2 │
└────────┘
┌────────┐
│Chart 3 │
└────────┘
┌────────┐
│Chart 4 │
└────────┘
┌────────┐
│Chart 5 │
└────────┘
┌────────┐
│Chart 6 │
└────────┘
┌────────┐
│Chart 7 │
└────────┘
```

## Color Scheme

### Chart Selector Container
- Background: White (#fff)
- Border: 2px solid purple-blue (#667eea)
- Border Radius: 10px
- Padding: 20px

### Info Box
- Background: Light blue (#f0f4ff)
- Text: Dark gray (#555)
- Icon: ℹ️ emoji

### Chart Buttons
- **Inactive**:
  - Background: Light gray (#f5f5f5)
  - Border: Light gray (#ddd)
  - Text: Black (#000)
  
- **Hover**:
  - Background: Light blue (#e3f2fd)
  - Border: Purple-blue (#667eea)
  - Transform: translateY(-2px)
  
- **Active**:
  - Background: Gradient (purple-blue #667eea to #764ba2)
  - Border: Purple-blue (#667eea)
  - Text: White (#fff)
  - Font Weight: 600 (semi-bold)

## Interaction Flow

### Step 1: Initial Display
```
User is in Phase B (Right Eye Refraction)
Chart selector appears automatically
First chart (200/150) is highlighted as active
```

### Step 2: User Clicks Chart 6 (20/20/20)
```
1. Button shows hover effect
2. Click event triggers
3. Frontend calls: POST /api/session/{id}/switch-chart
4. Request body: {"chart_index": 5}
5. Backend validates and switches chart
6. CURL command sent to phoropter
7. Response includes updated chart_info
```

### Step 3: UI Updates
```
1. Chart 1 (200/150) loses active styling
2. Chart 6 (20/20/20) gains active styling
3. Power display remains unchanged
4. Question and intents remain the same
5. History shows: "Switched to chart 6"
```

### Step 4: Continue Testing
```
User can now:
- Read the 20/20 chart
- Click "Able to read" to proceed
- Or click another chart to switch again
- Or click "Blurry" to add power
```

## Phase-Specific Behavior

### Phase A (Distance Vision)
```
Chart selector: HIDDEN
Only E-chart is shown
No chart switching available
```

### Phase B (Right/Left Eye Refraction)
```
Chart selector: VISIBLE ✓
All 7 Snellen charts available
Free switching between charts
Current chart highlighted
```

### JCC Phases
```
Chart selector: HIDDEN
JCC chart is fixed
No chart switching available
```

### Duochrome
```
Chart selector: HIDDEN
Duochrome chart is fixed
No chart switching available
```

## Error States

### Session Not Found
```
Alert: "No active session"
Chart buttons remain clickable but show error
```

### Invalid Phase
```
Chart selector automatically hidden
Switching not allowed in current phase
```

### Network Error
```
Alert: "Failed to switch chart. Please try again."
Previous chart remains active
User can retry
```

## Accessibility

### Keyboard Navigation
- Tab through chart buttons
- Enter/Space to activate
- Visual focus indicator

### Screen Readers
- Chart buttons have descriptive labels
- Active state announced
- Info text read first

### Touch Targets
- Minimum 44px × 44px touch area
- Adequate spacing between buttons
- Clear visual feedback on tap

## Animation & Transitions

### Chart Button Hover
```css
transition: all 0.3s ease
transform: translateY(-2px)
```

### Active State Change
```css
transition: background 0.3s ease
transition: color 0.3s ease
```

### Chart Selector Show/Hide
```css
display: none → display: block
No animation (instant)
```

## Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

## Performance

- Chart grid renders instantly
- No lag when switching charts
- Smooth hover animations
- Efficient DOM updates
