# 🎯 Eye Test Engine - Complete Frontend Guide

## What You Get

A **complete, production-ready web application** for conducting interactive eye tests with automatic phoropter control.

### Screenshots

```
┌─────────────────────────────────────────────────────────────┐
│  👁️ Eye Test Engine                                         │
│  Interactive Phoropter-Controlled Eye Examination           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📋 Phase: Distance Vision (Step 2.1)                       │
│                                                              │
│  ❓ Question                                                 │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Are you able to see big E clearly?                     │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  💬 Please select your response:                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ 1. Able to read                                        │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │ 2. Blurry                                              │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │ 3. Unable to read                                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Start the Backend

```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI
python -m eye_test_engine.api_server
```

You should see:
```
Starting Eye Test API Server...
Available endpoints:
  POST /api/session/start
  POST /api/session/<id>/respond
  GET  /api/session/<id>/status
  POST /api/session/<id>/end
 * Running on http://0.0.0.0:5000
```

### Step 2: Open the Frontend

**Option A: Use the Launcher Script (Easiest)**
```bash
cd eye_test_engine
./start_frontend.sh
```
This automatically starts both backend and frontend, and opens your browser!

**Option B: Manual**
```bash
cd eye_test_engine/frontend
python3 -m http.server 8080
# Then open http://localhost:8080 in your browser
```

**Option C: Direct File**
```bash
open eye_test_engine/frontend/index.html
```

### Step 3: Start Testing!

1. Click **"Start Eye Test"**
2. Answer each question by clicking an intent
3. Watch the phoropter update automatically
4. Complete all 10 phases
5. View your final prescription

---

## 🎨 Features

### 1. Beautiful Modern UI
- Gradient purple theme
- Smooth animations
- Responsive design
- Mobile-friendly

### 2. Real-Time Updates
- Live status indicator
- Current power display
- Phase tracking
- Response counter

### 3. Automatic Phoropter Control
- Sends curl commands automatically
- Updates charts
- Changes occluders
- Adjusts power

### 4. Interactive Intent Selection
- Click buttons
- Or use keyboard (1-9)
- Visual feedback
- Smooth transitions

### 5. Complete History Log
- Timestamps for each action
- Scrollable history
- Color-coded events
- Last 20 actions visible

### 6. Final Prescription Display
- Right eye (OD)
- Left eye (OS)
- SPH / CYL / AXIS
- Total responses

---

## 📋 Complete Test Flow

### Phase 1: Distance Vision
**Setup:** BINO + E-chart  
**Question:** "Are you able to see big E clearly?"  
**Intents:**
- Able to read
- Blurry
- Unable to read

**Phoropter Actions:**
```bash
# Reset to 0/0/180
curl -X POST .../reset

# Display E-chart
curl -X POST .../run-tests -d '{"test_cases":[{"chart":{"tab":"Chart1","chart_items":["chart_9"]}}]}'
```

---

### Phase 2: Right Eye Refraction
**Setup:** Left_Occluded + Snellen  
**Question:** "I'm covering your left eye. Please read the line you can see clearly."  
**Intents:**
- Able to read
- Blurry
- Unable to read
- Getting better

**Phoropter Actions:**
```bash
# Display Snellen 20/20/20
curl -X POST .../run-tests -d '{"test_cases":[{"chart":{"tab":"Chart1","chart_items":["chart_15"]}}]}'

# Occlude left eye
curl -X POST .../run-tests -d '{"test_cases":[{"aux_lens":"AuxLensL","right_eye":{"sph":0.0,"cyl":0.0,"axis":180},"left_eye":{"sph":0.0,"cyl":0.0,"axis":180}}]}'
```

---

### Phase 3: JCC Axis (Right Eye)
**Setup:** Right_Axis_Flip1/2 + JCC Chart  
**Questions:**
- Flip 1: "Focus on the dot chart. Is this better? (Flip 1)"
- Flip 2: "Or is this better? (Flip 2)"

**Intents:**
- Flip 1: GAP Axis (increase axis by 5°)
- Flip 2: RAM Axis (decrease axis by 5°)
- Both Same (no change needed)

**Phoropter Actions:**
```bash
# Display JCC chart
curl -X POST .../run-tests -d '{"test_cases":[{"chart":{"tab":"Chart1","chart_items":["chart_19"]}}]}'

# Set to right eye mode
curl -X POST .../run-tests -d '{"test_cases":[{"jcc":"R"}]}'

# Flip handle
curl -X POST .../run-tests -d '{"test_cases":[{"jcc":"handle"}]}'
```

---

### Phase 4: JCC Power (Right Eye)
**Setup:** Right_Power_Flip1/2 + JCC Chart  
**Questions:** Same as Axis  
**Intents:**
- Flip 1: GAP Power (increase cylinder by 0.25D)
- Flip 2: RAM Power (decrease cylinder by 0.25D)
- Both Same (no change needed)

**Phoropter Actions:**
```bash
# Switch to power mode
curl -X POST .../run-tests -d '{"test_cases":[{"jcc":"power_axis_switch"}]}'

# Increase/Decrease
curl -X POST .../run-tests -d '{"test_cases":[{"jcc":"increase"}]}'
curl -X POST .../run-tests -d '{"test_cases":[{"jcc":"decrease"}]}'
```

---

### Phase 5: Duochrome (Right Eye)
**Setup:** Left_Occluded + Duochrome  
**Question:** "Which is clearer: red or green, or are they the same?"  
**Intents:**
- Red
- Green
- Both Same

**Phoropter Actions:**
```bash
# Display duochrome
curl -X POST .../run-tests -d '{"test_cases":[{"chart":{"tab":"Chart1","chart_items":["chart_17"]}}]}'
```

---

### Phases 6-9: Left Eye
Same sequence as right eye:
- Phase 6: Left Eye Refraction (Right_Occluded)
- Phase 7: JCC Axis Left
- Phase 8: JCC Power Left
- Phase 9: Duochrome Left

---

### Phase 10: Binocular Balance
**Setup:** BINO + Snellen  
**Question:** "Please read the line you can see clearly."  
**Intents:**
- Able to read
- Blurry
- Unable to read

**Phoropter Actions:**
```bash
# Remove occluder
curl -X POST .../run-tests -d '{"test_cases":[{"aux_lens":"OFF",...}]}'
```

---

## 🔧 Technical Details

### Frontend Stack
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with gradients
- **Vanilla JavaScript** - No frameworks needed
- **Fetch API** - For backend communication

### Backend Stack
- **Flask** - Python web framework
- **Flask-CORS** - Cross-origin support
- **PyYAML** - Configuration files
- **Custom State Machine** - Phase logic

### API Communication

```javascript
// Start session
fetch('http://localhost:5000/api/session/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: 'session_123' })
})

// Submit response
fetch('http://localhost:5000/api/session/session_123/respond', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ intent: 'Able to read' })
})
```

### Phoropter Communication

```javascript
// Reset phoropter
fetch('https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/reset', {
    method: 'POST'
})

// Set chart
fetch('https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        test_cases: [{
            chart: { tab: "Chart1", chart_items: ["chart_15"] }
        }]
    })
})
```

---

## 🎯 User Experience Flow

### 1. Welcome Screen
```
┌────────────────────────────────────┐
│ Welcome to the Eye Test            │
│                                    │
│ This interactive system will...    │
│                                    │
│ [Start Eye Test]                   │
└────────────────────────────────────┘
```

### 2. Active Test
```
┌────────────────────────────────────┐
│ Phase: Distance Vision             │
│                                    │
│ Question: Are you able to...       │
│                                    │
│ Intents:                           │
│ [1. Able to read]                  │
│ [2. Blurry]                        │
│ [3. Unable to read]                │
│                                    │
│ [End Test]                         │
└────────────────────────────────────┘
```

### 3. Loading State
```
┌────────────────────────────────────┐
│ [Spinner Animation]                │
│ Processing...                      │
└────────────────────────────────────┘
```

### 4. Completion Screen
```
┌────────────────────────────────────┐
│ ✅ Test Complete!                  │
│                                    │
│ Final Prescription:                │
│ Right Eye: -1.25 / -0.50 / 90°    │
│ Left Eye:  -1.00 / -0.75 / 180°   │
│                                    │
│ [Start New Test]                   │
└────────────────────────────────────┘
```

---

## 📊 Status Panel

### Session Status
- **Active Indicator** - Green dot when running
- **Session ID** - Unique identifier
- **Current Phase** - Which phase you're in
- **Response Count** - Number of answers given

### Current Power
- **Right Eye** - SPH / CYL / AXIS
- **Left Eye** - SPH / CYL / AXIS
- **Occluder** - BINO / Left_Occluded / Right_Occluded
- **Chart** - Current chart being displayed

### Test History
- Scrollable log
- Timestamps
- Color-coded events
- Last 20 actions

---

## ⌨️ Keyboard Shortcuts

- **1-9** - Select intent by number
- **Ctrl+C** - Stop servers (in terminal)

---

## 🔍 Troubleshooting

### "Failed to start test"
**Cause:** Backend not running  
**Fix:** Run `python -m eye_test_engine.api_server`

### "Could not update phoropter"
**Cause:** Network issue or wrong API endpoint  
**Fix:** Check phoropter API URL in `app.js`

### CORS Errors
**Cause:** Opening HTML file directly  
**Fix:** Use `python3 -m http.server 8080`

### Session Not Found
**Cause:** Backend restarted  
**Fix:** Refresh page and start new test

---

## 🚀 Production Deployment

### Backend
```bash
# Install gunicorn
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:5000 eye_test_engine.api_server:app
```

### Frontend
```bash
# Serve with nginx
server {
    listen 80;
    root /path/to/eye_test_engine/frontend;
    index index.html;
}
```

### Environment Variables
```bash
export PHOROPTER_URL="https://your-api.com"
export PHOROPTER_ID="your-id"
export FLASK_ENV="production"
```

---

## 📝 Customization

### Change Colors
Edit `index.html` CSS:
```css
/* Main gradient */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);

/* Accent */
color: #667eea;
border: 2px solid #667eea;
```

### Change Questions
Edit `eye_test_engine/config/protocol.yaml`:
```yaml
phases:
  distance_vision:
    questions:
      - "Your custom question here"
```

### Add New Phase
Edit `interactive_session.py`:
```python
def _determine_next_phase(self, intent: str) -> str:
    phase_flow = {
        "distance_vision": "your_new_phase",
        "your_new_phase": "right_eye_refraction",
        # ...
    }
```

---

## 📚 Files Reference

```
eye_test_engine/
├── frontend/
│   ├── index.html          # Main UI (HTML + CSS)
│   ├── app.js              # Frontend logic
│   └── README.md           # Frontend docs
├── api_server.py           # Flask backend
├── interactive_session.py  # Session orchestrator
├── start_frontend.sh       # Launcher script
└── FRONTEND_GUIDE.md       # This file
```

---

## ✅ Testing Checklist

- [ ] Backend starts without errors
- [ ] Frontend loads in browser
- [ ] "Start Eye Test" button works
- [ ] Questions display correctly
- [ ] Intent buttons are clickable
- [ ] Keyboard shortcuts work (1-9)
- [ ] Status panel updates
- [ ] History log shows events
- [ ] Phoropter commands execute
- [ ] All 10 phases complete
- [ ] Final prescription displays
- [ ] "Start New Test" works

---

## 🎓 Next Steps

1. ✅ **Test locally** - Run through a complete test
2. ✅ **Customize** - Change colors, questions, flow
3. ✅ **Deploy** - Set up production environment
4. ✅ **Integrate** - Connect to your systems
5. ✅ **Monitor** - Add logging and analytics

---

## 🆘 Support

- **Main Docs:** `eye_test_engine/README.md`
- **API Guide:** `eye_test_engine/API_USAGE.md`
- **Quick Start:** `eye_test_engine/QUICK_START.md`
- **State Machine:** `docs/STATE_MACHINE_DIAGRAM.md`

---

**Ready to start? Run `./start_frontend.sh` and begin testing!** 🎉
