# Eye Test Engine — Frontend & UI Guide

## Overview

A production-ready web application for interactive eye tests with automatic phoropter control. The frontend guides patients through 10 phases, automatically controls phoropter hardware via the API, and displays real-time status updates.

---

## Quick Start

### Option 1: One-Click Launch

```bash
cd eye_test_engine
./start_frontend.sh
```

Automatically starts backend (port 5000), frontend server (port 8080), and opens the browser.

### Option 2: Manual

```bash
# Terminal 1
python -m eye_test_engine.api_server

# Terminal 2
cd eye_test_engine/frontend && python3 -m http.server 8080

# Browser
open http://localhost:8080
```

### Option 3: Direct File

```bash
open eye_test_engine/frontend/index.html
```

---

## UI Layout

```
┌───────────────────────────────────────────────────────────────┐
│ HEADER                                                         │
│ Eye Test Engine    [Set AR] [Set Lenso] [Compare]  Apply:[AR][Lenso]  Device ID: [phoropter-1] │
└───────────────────────────────────────────────────────────────┘

┌─────────────────────────┬─────────────────────────────────────┐
│ TEST PANEL              │ INFO PANEL                           │
│                         │                                      │
│ Phase: Phase B          │ Status: Active                       │
│                         │ Session ID: session_123              │
│ Question: I'm covering  │ Current Phase: Phase B               │
│ your left eye...        │ Responses: 5                         │
│                         │                                      │
│ [1. Able to read]      │ Right: -1.25 / -0.50 / 180°         │
│ [2. Blurry]            │ Left:  -1.00 / -0.75 / 90°          │
│ [3. Unable to read]    │ Occluder: Left_Occluded              │
│                         │ Chart: snellen_chart_20_20_20        │
│ 📊 Chart Selection      │                                      │
│ [chart grid...]        │ History log                          │
└─────────────────────────┴─────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│ FOOTER: Jump to Phase: [Select Phase ▼] [Go]                 │
└───────────────────────────────────────────────────────────────┘
```

### Color Scheme

- Primary gradient: Purple `#667eea` → `#764ba2`
- Background: White with soft shadows
- Active button: Gradient purple-blue, white text
- Disabled button: Gray, 40% opacity

---

## Power Controls

### Overview

Power entry and application are separate. Store AR/Lenso values first, then apply them with dedicated buttons.

### Workflow

1. **Store:** Click "Set AR" or "Set Lenso" → enter SPH/CYL/AXIS for both eyes → "Save"
2. **Apply:** Click "AR" or "Lenso" apply button in header

### Button States

| Condition | AR Button | Lenso Button |
|-----------|-----------|--------------|
| No power stored | Disabled | Disabled |
| AR stored | **Enabled** | Disabled |
| Both stored | **Enabled** | **Enabled** |
| AR applied | Active (highlighted) | Enabled |

### Validation

All 6 fields (SPH, CYL, AXIS × 2 eyes) must be filled before "Save" activates.

### Power Storage

```javascript
storedPower = {
    ar:    { right: {sph, cyl, axis}, left: {sph, cyl, axis} },
    lenso: { right: {sph, cyl, axis}, left: {sph, cyl, axis} }
}
currentAppliedPower = 'none'  // 'ar' | 'lenso' | 'none'
```

---

## Device ID (Phoropter ID)

The **Device ID** input in the header controls which phoropter receives all API calls. It defaults to `phoropter-1` and persists across page refreshes via `localStorage`.

- Change the value and tab away to save it
- All `CONFIG.phoropterId` references in `app.js` read this input at call time (getter pattern)

---

## Processing State

When a user clicks an intent:
1. All intent buttons are hidden immediately
2. "Processing…" message shown
3. Backend processes, phoropter updates
4. New intents appear only when ready

This prevents double-clicks and eliminates button flicker.

---

## Auto-Flip (JCC Phases)

During JCC phases with auto-flip:
1. Flip 1 shown — buttons hidden
2. Countdown: "⏱️ Showing Flip 2 in X seconds…"
3. `AUTO_FLIP` sent to backend after countdown
4. Flip 2 shown — 4 intent buttons enabled

```javascript
async function handleAutoFlip(waitSeconds) {
    intentButtons.forEach(btn => btn.disabled = true);
    for (let i = waitSeconds; i > 0; i--) {
        countdownDiv.textContent = `⏱️ Showing Flip 2 in ${i} second${i > 1 ? 's' : ''}...`;
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    const data = await fetch(`/api/session/${id}/respond`, {
        method: 'POST',
        body: JSON.stringify({ intent: 'AUTO_FLIP' })
    }).then(r => r.json());
    displayQuestion(data);
}
```

---

## Complete Test Flow (10 Phases)

| Phase | Setup | Question |
|-------|-------|---------|
| A: Distance Vision | BINO + E-chart | "Are you able to see big E clearly?" |
| B: Right Eye Refraction | Left_Occluded + Snellen | "I'm covering your left eye. Please read the line…" |
| E: JCC Axis Right | JCC chart, Flip1/2 | "Which flip was better?" |
| F: JCC Power Right | JCC chart, Flip1/2 | "Which flip was better?" |
| G: Duochrome Right | Left_Occluded + Duochrome | "Which is clearer: red or green?" |
| D: Left Eye Refraction | Right_Occluded + Snellen | "I'm covering your right eye. Please read the line…" |
| H: JCC Axis Left | JCC chart, Flip1/2 | "Which flip was better?" |
| I: JCC Power Left | JCC chart, Flip1/2 | "Which flip was better?" |
| J: Duochrome Left | Right_Occluded + Duochrome | "Which is clearer: red or green?" |
| K: Binocular Balance | BINO + chart_20 | "Which line is less blurry?" |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| 1–9 | Select intent by number |
| F12 | Open browser DevTools |
| F5 (VSCode) | Continue debugging |
| F10 (VSCode) | Step Over |
| F11 (VSCode) | Step Into |

---

## Debugging Guide

### Chrome DevTools (Fastest)

```bash
cd eye_test_engine/frontend && python -m http.server 8000
# Open http://localhost:8000, press F12 → Sources tab
```

### VSCode Debugger (Best)

1. Install **Debugger for Chrome** extension
2. Start local server on port 8000
3. Press `F5` → Select "Debug Chrome Frontend"
4. Set breakpoints directly in VSCode

### Key Functions to Debug

| Function | Purpose |
|---------|---------|
| `startTest()` | Session initialization |
| `submitIntent()` | Response handling |
| `setPhoropter()` | Hardware communication |
| `displayQuestion()` | UI updates |
| `handleAutoFlip()` | JCC timing logic |

### Watch Expressions (VSCode)

```
sessionState.sessionId
sessionState.responseCount
sessionState.currentPhase
CONFIG.backendUrl
CONFIG.phoropterId
```

### Backend Debugging (VS Code)

Two methods:
1. **Launch with debugger:** Start Flask with debugpy on port 5678
2. **Attach to running:** Run `python -m debugpy --listen 5678 api_server.py`, then attach

---

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| "Failed to start test" | Backend not running | `python -m eye_test_engine.api_server` |
| "Could not update phoropter" | Wrong API endpoint | Check `CONFIG.phoropterUrl` in `app.js` |
| CORS Errors | Opening HTML file directly | Use `python3 -m http.server 8080` |
| Session Not Found | Backend restarted | Refresh page, start new test |
| Breakpoints not hit | Debugger not attached | Verify console shows "Debugger listening on 5678" |

---

## Customization

### Change Colors

```css
/* index.html */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```

### Change Questions

```yaml
# eye_test_engine/config/protocol.yaml
phases:
  distance_vision:
    questions:
      - "Your custom question here"
```

---

## Production Deployment

```bash
# Backend
gunicorn -w 4 -b 0.0.0.0:5000 eye_test_engine.api_server:app

# Frontend (nginx example)
server {
    listen 80;
    root /path/to/eye_test_engine/frontend;
    index index.html;
}
```

---

## Testing Checklist

- [ ] Backend starts without errors
- [ ] "Start Eye Test" button works
- [ ] Intent buttons are clickable
- [ ] Keyboard shortcuts work (1–9)
- [ ] Status panel updates correctly
- [ ] History log shows events
- [ ] All 10 phases complete
- [ ] Final prescription displays
- [ ] Power modals save correctly
- [ ] Apply buttons enable/disable correctly
- [ ] Processing state hides buttons
- [ ] Auto-flip countdown works
- [ ] Device ID persists across refreshes

---

## File Reference

```
eye_test_engine/
├── frontend/
│   ├── index.html          # UI (HTML + CSS)
│   └── app.js              # Frontend logic
├── api_server.py           # Flask backend
├── interactive_session.py  # Session orchestrator
├── start_frontend.sh       # One-click launcher
└── config/
    └── protocol.yaml       # Phase configuration
```

**Status:** ✅ Production-ready — all features implemented and tested
