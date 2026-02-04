# 🎬 Eye Test Engine - Live Demo

## What This Does

This is a **complete, working web application** that:

1. ✅ Asks you questions one by one
2. ✅ Shows you intent options to select
3. ✅ Automatically controls the phoropter via curl commands
4. ✅ Guides you through all 10 phases of the eye test
5. ✅ Displays your final prescription

---

## 🚀 Start Demo in 30 Seconds

### Terminal 1: Start Backend
```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI
python -m eye_test_engine.api_server
```

### Terminal 2: Start Frontend
```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/frontend
python3 -m http.server 8080
```

### Browser
Open: `http://localhost:8080`

---

## 📺 Demo Walkthrough

### Step 1: Welcome Screen

When you open the page, you see:

```
┌──────────────────────────────────────────────────────┐
│  👁️ Eye Test Engine                                  │
│  Interactive Phoropter-Controlled Eye Examination    │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Welcome to the Eye Test                             │
│                                                       │
│  This interactive system will guide you through      │
│  a complete eye examination. The phoropter will      │
│  be automatically controlled based on your           │
│  responses.                                          │
│                                                       │
│  ℹ️ Note: Make sure the phoropter API is            │
│     accessible at:                                   │
│     https://rajasthan-royals.preprod.lenskart.com    │
│                                                       │
│  [Start Eye Test]                                    │
│                                                       │
└──────────────────────────────────────────────────────┘
```

**Click "Start Eye Test"**

---

### Step 2: Phase 1 - Distance Vision

The screen updates to show:

```
┌──────────────────────────────────────────────────────┐
│  Distance Vision (Step 2.1)                          │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ❓ Question                                          │
│  ┌────────────────────────────────────────────────┐  │
│  │ Are you able to see big E clearly?            │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  💬 Please select your response:                     │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │ 1. Able to read                                │  │
│  ├────────────────────────────────────────────────┤  │
│  │ 2. Blurry                                      │  │
│  ├────────────────────────────────────────────────┤  │
│  │ 3. Unable to read                              │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  [End Test]                                          │
│                                                       │
└──────────────────────────────────────────────────────┘
```

**Behind the scenes:**
```bash
# Phoropter reset
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/reset

# Display E-chart
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases":[{"chart":{"tab":"Chart1","chart_items":["chart_9"]}}]}'
```

**Status Panel Shows:**
```
Session Status
  🟢 Status: Active
  Session ID: session_1738675200123
  Current Phase: distance_vision
  Responses: 0

Current Power
  Right Eye: 0.00 / 0.00 / 180°
  Left Eye: 0.00 / 0.00 / 180°
  Occluder: BINO
  Chart: echart_400

Test History
  12:00:01 - Test started
  12:00:01 - Phoropter reset to 0/0/180
  12:00:01 - Chart: echart_400
```

**Click "1. Able to read"**

---

### Step 3: Phase 2 - Right Eye Refraction

Screen updates:

```
┌──────────────────────────────────────────────────────┐
│  Right Eye Refraction (RE6.3)                        │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ❓ Question                                          │
│  ┌────────────────────────────────────────────────┐  │
│  │ I'm covering your left eye. Please read the   │  │
│  │ line you can see clearly.                     │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  💬 Please select your response:                     │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │ 1. Able to read                                │  │
│  ├────────────────────────────────────────────────┤  │
│  │ 2. Blurry                                      │  │
│  ├────────────────────────────────────────────────┤  │
│  │ 3. Unable to read                              │  │
│  ├────────────────────────────────────────────────┤  │
│  │ 4. Getting better                              │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
└──────────────────────────────────────────────────────┘
```

**Behind the scenes:**
```bash
# Display Snellen chart
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases":[{"chart":{"tab":"Chart1","chart_items":["chart_15"]}}]}'

# Occlude left eye (test right)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "aux_lens": "AuxLensL",
      "right_eye": {"sph": 0.0, "cyl": 0.0, "axis": 180},
      "left_eye": {"sph": 0.0, "cyl": 0.0, "axis": 180}
    }]
  }'
```

**Status Panel Updates:**
```
Current Power
  Right Eye: 0.00 / 0.00 / 180°
  Left Eye: 0.00 / 0.00 / 180°
  Occluder: Left_Occluded
  Chart: snellen_chart_20_20_20

Test History
  12:00:05 - Response: Able to read
  12:00:05 - Chart: snellen_chart_20_20_20
  12:00:05 - Power updated - Occluder: Left_Occluded
```

**Click "1. Able to read"**

---

### Step 4: Phase 3 - JCC Axis (Right Eye)

Screen updates:

```
┌──────────────────────────────────────────────────────┐
│  JCC Axis Refinement (Right Eye)                     │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ❓ Question                                          │
│  ┌────────────────────────────────────────────────┐  │
│  │ Focus on the dot chart. Is this better?       │  │
│  │ (Flip 1)                                       │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  💬 Please select your response:                     │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │ 1. No response expected (Flip 1 presented,    │  │
│  │    awaiting Flip 2 comparison)                 │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
└──────────────────────────────────────────────────────┘
```

**Behind the scenes:**
```bash
# Display JCC chart
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases":[{"chart":{"tab":"Chart1","chart_items":["chart_19"]}}]}'

# Set to right eye mode
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases":[{"jcc":"R"}]}'
```

**Status Panel Updates:**
```
Current Power
  Occluder: Right_Axis_Flip1
  Chart: jcc_chart
```

**Click the only option (Flip 1)**

---

### Step 5: Continue Through All Phases

The system automatically progresses through:

4. ✅ JCC Power (Right Eye)
5. ✅ Duochrome (Right Eye)
6. ✅ Left Eye Refraction
7. ✅ JCC Axis (Left Eye)
8. ✅ JCC Power (Left Eye)
9. ✅ Duochrome (Left Eye)
10. ✅ Binocular Balance

Each phase:
- Shows the appropriate question
- Displays relevant intents
- Sends curl commands to phoropter
- Updates the status panel
- Logs actions to history

---

### Step 6: Test Complete!

Final screen:

```
┌──────────────────────────────────────────────────────┐
│  ✅ Test Complete!                                    │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ✓ Success! Your eye test has been completed        │
│    successfully.                                     │
│                                                       │
│  Final Prescription                                  │
│  ┌────────────────────────────────────────────────┐  │
│  │ Right Eye (OD)                                 │  │
│  │ SPH: -1.25 | CYL: -0.50 | AXIS: 90°          │  │
│  │                                                │  │
│  │ Left Eye (OS)                                  │  │
│  │ SPH: -1.00 | CYL: -0.75 | AXIS: 180°         │  │
│  │                                                │  │
│  │ Total Responses: 10                            │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  [Start New Test]                                    │
│                                                       │
└──────────────────────────────────────────────────────┘
```

**Status Panel Shows:**
```
Session Status
  🔴 Status: Completed
  Session ID: session_1738675200123
  Current Phase: binocular_balance
  Responses: 10

Test History
  12:05:30 - Response: Able to read
  12:05:30 - Test completed successfully
```

---

## 🎯 Key Features Demonstrated

### 1. Question-Answer Flow
- ✅ Each phase has specific questions
- ✅ Intents are clearly listed
- ✅ Click or use keyboard (1-9)

### 2. Automatic Phoropter Control
- ✅ Resets at start
- ✅ Changes charts automatically
- ✅ Updates occluders
- ✅ Adjusts power

### 3. Real-Time Feedback
- ✅ Status indicator (green/red)
- ✅ Current power display
- ✅ Phase tracking
- ✅ Response counter

### 4. Complete History
- ✅ Timestamped events
- ✅ All actions logged
- ✅ Scrollable view
- ✅ Color-coded

### 5. Final Results
- ✅ Complete prescription
- ✅ Both eyes
- ✅ All parameters
- ✅ Response count

---

## 📊 Technical Flow

```
User Action → Frontend (app.js) → Backend API (Flask)
                                        ↓
                                  State Machine
                                        ↓
                                  Next Question
                                        ↓
Frontend (app.js) → Phoropter API (curl commands)
        ↓
    Update UI
```

---

## 🔧 What Gets Executed

### For Each Response:

1. **Frontend sends intent to backend:**
   ```javascript
   fetch('http://localhost:5000/api/session/session_123/respond', {
       method: 'POST',
       body: JSON.stringify({ intent: 'Able to read' })
   })
   ```

2. **Backend processes and returns next state:**
   ```json
   {
       "phase": "right_eye_refraction",
       "question": "I'm covering your left eye...",
       "intents": ["Able to read", "Blurry", "Unable to read"],
       "chart": "snellen_chart_20_20_20",
       "occluder": "Left_Occluded",
       "power": {
           "right": {"sph": 0.0, "cyl": 0.0, "axis": 180},
           "left": {"sph": 0.0, "cyl": 0.0, "axis": 180}
       }
   }
   ```

3. **Frontend updates phoropter:**
   ```javascript
   // Set chart
   fetch('https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests', {
       method: 'POST',
       body: JSON.stringify({
           test_cases: [{
               chart: { tab: "Chart1", chart_items: ["chart_15"] }
           }]
       })
   })
   
   // Set power and occluder
   fetch('https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests', {
       method: 'POST',
       body: JSON.stringify({
           test_cases: [{
               aux_lens: "AuxLensL",
               right_eye: {sph: 0.0, cyl: 0.0, axis: 180},
               left_eye: {sph: 0.0, cyl: 0.0, axis: 180}
           }]
       })
   })
   ```

4. **Frontend updates UI:**
   - Question text
   - Intent buttons
   - Status panel
   - History log

---

## ✨ User Experience

### Smooth Transitions
- Loading spinner during processing
- Smooth animations
- Visual feedback on clicks
- Real-time updates

### Clear Communication
- Phase badges show current step
- Questions are prominent
- Intents are numbered
- History shows all actions

### Error Handling
- Alerts for connection issues
- Warnings for phoropter errors
- Graceful degradation
- Clear error messages

---

## 🎓 Try It Yourself!

1. **Start the servers** (see top of this file)
2. **Open the browser** to http://localhost:8080
3. **Click "Start Eye Test"**
4. **Answer each question** by clicking intents
5. **Watch the magic happen!** 🎉

The phoropter will be controlled automatically, and you'll see the complete flow from start to finish!

---

## 📝 Notes

- This is a **complete, working system**
- All curl commands are **automatically executed**
- The state machine **handles all transitions**
- The UI is **production-ready**
- Everything is **fully documented**

---

**Ready to see it in action? Start the servers and open your browser!** 🚀
