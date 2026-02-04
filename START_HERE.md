# 🚀 START HERE - Eye Test Engine Frontend

## What This Is

A **complete web application** that:
1. ✅ Asks questions one by one
2. ✅ Shows intent options to select
3. ✅ Automatically controls the phoropter via curl commands
4. ✅ Guides through all 10 phases of the eye test
5. ✅ Displays final prescription

---

## Quick Start (30 Seconds)

### Option 1: One-Click Launch (Easiest)
```bash
cd eye_test_engine
./start_frontend.sh
```
This automatically starts everything and opens your browser!

### Option 2: Manual Launch
```bash
# Terminal 1: Backend
python -m eye_test_engine.api_server

# Terminal 2: Frontend
cd eye_test_engine/frontend
python3 -m http.server 8080

# Browser
open http://localhost:8080
```

---

## What You'll See

### 1. Welcome Screen
Click "Start Eye Test" to begin

### 2. Interactive Questions
- Question displayed clearly
- Intent options as clickable buttons
- Use mouse or keyboard (1-9)

### 3. Automatic Phoropter Control
Behind the scenes, curl commands are sent:
- Reset phoropter
- Change charts
- Update occluders
- Adjust power

### 4. Real-Time Status
- Current phase
- Power display
- History log
- Response count

### 5. Final Prescription
- Right eye (OD)
- Left eye (OS)
- Complete SPH/CYL/AXIS

---

## 10 Test Phases

1. Distance Vision (BINO + E-chart)
2. Right Eye Refraction (Left_Occluded + Snellen)
3. JCC Axis Right (Flip1/Flip2)
4. JCC Power Right (Flip1/Flip2)
5. Duochrome Right (Red/Green)
6. Left Eye Refraction (Right_Occluded + Snellen)
7. JCC Axis Left (Flip1/Flip2)
8. JCC Power Left (Flip1/Flip2)
9. Duochrome Left (Red/Green)
10. Binocular Balance (BINO + Snellen)

---

## Documentation

- **COMPLETE_FRONTEND_DELIVERY.md** - Full delivery summary
- **eye_test_engine/FRONTEND_GUIDE.md** - Complete usage guide
- **eye_test_engine/frontend/README.md** - Frontend docs
- **eye_test_engine/frontend/DEMO.md** - Step-by-step demo
- **eye_test_engine/API_USAGE.md** - curl command reference
- **eye_test_engine/QUICK_START.md** - Quick reference

---

## File Structure

```
eye_test_engine/
├── frontend/
│   ├── index.html          # Beautiful UI
│   ├── app.js              # Application logic
│   ├── README.md           # Frontend docs
│   └── DEMO.md             # Demo walkthrough
├── api_server.py           # Flask backend
├── interactive_session.py  # Session orchestrator
├── start_frontend.sh       # Launcher script
├── FRONTEND_GUIDE.md       # Usage guide
└── config/
    └── protocol.yaml       # Phase configuration
```

---

## Requirements

- Python 3.7+
- Flask
- Flask-CORS
- PyYAML

Install dependencies:
```bash
pip install flask flask-cors pyyaml
```

---

## Troubleshooting

### "Failed to start test"
**Fix:** Make sure backend is running: `python -m eye_test_engine.api_server`

### "Could not update phoropter"
**Fix:** Check phoropter API URL in `eye_test_engine/frontend/app.js`

### CORS Errors
**Fix:** Use `python3 -m http.server 8080` instead of opening file directly

---

## Next Steps

1. ✅ **Start the servers** (see Quick Start above)
2. ✅ **Open browser** to http://localhost:8080
3. ✅ **Click "Start Eye Test"**
4. ✅ **Answer questions** by clicking intents
5. ✅ **Complete all 10 phases**
6. ✅ **View final prescription**

---

## Support

- Read **COMPLETE_FRONTEND_DELIVERY.md** for full details
- Check **eye_test_engine/FRONTEND_GUIDE.md** for troubleshooting
- See **eye_test_engine/frontend/DEMO.md** for step-by-step walkthrough

---

**Ready? Run `./eye_test_engine/start_frontend.sh` and start testing!** 🎉
