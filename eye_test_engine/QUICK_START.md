# Eye Test Engine - Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. View the Conversation Flow
```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI
python -m eye_test_engine.demo_conversation
```

**What you'll see:**
- Questions for each phase
- Patient response options (intents)
- Machine state requirements

---

### 2. Process Existing Test Data
```bash
python -m eye_test_engine.run Curated_Conversations/ --output results/ --summary
```

**Output:**
- `results/annotated_*.csv` - Original CSV + Phase_ID + Phase_Name columns
- `results/summary_*.txt` - Final prescription and phase distribution

---

### 3. Run Interactive Test (with API)
```bash
# Terminal 1: Start API server
python -m eye_test_engine.api_server

# Terminal 2: Start a session
curl -X POST http://localhost:5000/api/session/start \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test_001"}'

# Submit patient response
curl -X POST http://localhost:5000/api/session/test_001/respond \
  -H "Content-Type: application/json" \
  -d '{"intent": "Able to read"}'
```

---

## 📋 Phase Cheat Sheet

| Phase | Question | Intents | Exit Condition |
|-------|----------|---------|----------------|
| **Distance Vision** | "Please read the line you can see clearly." | Able to read, Blurry, Unable to read | Baseline established |
| **Right Eye Refraction** | "I'm covering your left eye. Please read the line..." | Able to read, Blurry, Unable to read, Getting better | 2x "Unable to read" after SPH |
| **JCC Axis Right** | Flip1: "Is this better?" Flip2: "Or is this better?" | GAP Axis, RAM Axis, Both Same | Axis stable |
| **JCC Power Right** | Flip1: "Is this better?" Flip2: "Or is this better?" | GAP Power, RAM Power, Both Same | Power stable or 0.00 |
| **Duochrome Right** | "Which is clearer: red or green, or are they the same?" | Red, Green, Both Same | Response received |
| **Left Eye Refraction** | "I'm covering your right eye. Please read the line..." | Able to read, Blurry, Unable to read, Getting better | 2x "Unable to read" after SPH |
| **JCC Axis Left** | Flip1: "Is this better?" Flip2: "Or is this better?" | GAP Axis, RAM Axis, Both Same | Axis stable |
| **JCC Power Left** | Flip1: "Is this better?" Flip2: "Or is this better?" | GAP Power, RAM Power, Both Same | Power stable or 0.00 |
| **Duochrome Left** | "Which is clearer: red or green, or are they the same?" | Red, Green, Both Same | Response received |
| **Binocular Balance** | "Please read the line you can see clearly." | Able to read, Blurry, Unable to read | Balance verified |

---

## 🔧 Phoropter Control (Quick Reference)

### Reset
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/reset
```

### Set Power
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{
    "test_cases": [{
      "aux_lens": "AuxLensL",
      "right_eye": {"sph": -1.0, "cyl": -0.5, "axis": 90},
      "left_eye": {"sph": -0.75, "cyl": -0.25, "axis": 180}
    }]
  }'
```

### Display Chart
```bash
# Snellen 20/20/20
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_15"]}}]}'

# JCC Chart
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_19"]}}]}'

# Duochrome
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"chart": {"tab": "Chart1", "chart_items": ["chart_17"]}}]}'
```

### JCC Operations
```bash
# Handle (flip)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "handle"}]}'

# Increase (GAP)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "increase"}]}'

# Decrease (RAM)
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"jcc": "decrease"}]}'
```

---

## 📁 File Structure

```
eye_test_engine/
├── run.py                      # ⚡ Batch processing
├── interactive_session.py      # 💬 Live session
├── api_server.py               # 🌐 API endpoints
├── demo_conversation.py        # 📺 Demo flow
├── config/
│   ├── protocol.yaml           # 📋 Phase config
│   └── thresholds.yaml         # ⚙️ Parameters
├── core/
│   ├── context.py              # 📊 Data structures
│   └── state_machine.py        # 🤖 Logic
└── modules/
    ├── spherical.py            # 🔵 Sphere
    ├── cylinder_axis.py        # 🔄 Axis
    ├── cylinder_power.py       # ⚡ Power
    ├── duochrome.py            # 🔴🟢 Red/Green
    └── binocular_balance.py    # 👁️👁️ Balance
```

---

## 🎯 Common Tasks

### Analyze a single CSV
```bash
python -m eye_test_engine.run path/to/file.csv --summary
```

### Analyze all CSVs in a directory
```bash
python -m eye_test_engine.run Curated_Conversations/ --output results/
```

### Change thresholds
Edit `eye_test_engine/config/thresholds.yaml`:
```yaml
sphere_refinement:
  unable_read_threshold: 2  # Change to 3 for more attempts

cylinder_refinement:
  axis_increment: 5  # Change to 10 for larger steps
  power_increment: 0.25  # Change to 0.50 for larger steps
```

### Customize questions
Edit `eye_test_engine/config/protocol.yaml`:
```yaml
phases:
  distance_vision:
    questions:
      - "Your custom question here"
    intents:
      - "Custom intent 1"
      - "Custom intent 2"
```

---

## 📚 Documentation

- `README.md` - Full documentation
- `API_USAGE.md` - curl command examples
- `SUMMARY.md` - Complete overview
- `docs/STATE_MACHINE_DIAGRAM.md` - State machine diagram
- `docs/CURATION_LOGIC.md` - Snellen/JCC logic
- `refined_clinical_protocol.txt` - Clinical mapping rules

---

## ✅ Verification

Test that everything works:
```bash
# 1. Demo runs without errors
python -m eye_test_engine.demo_conversation

# 2. Batch processing works
python -m eye_test_engine.run Curated_Conversations/ --output test_output/ --summary

# 3. Check output
ls test_output/
cat test_output/summary_*.txt | head -30
```

---

## 🆘 Troubleshooting

**Problem:** Module not found  
**Solution:** Run from project root: `cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI`

**Problem:** No CSVs in Curated_Conversations  
**Solution:** Run `python filter_valid_csvs.py && python curate_conversations.py`

**Problem:** API server won't start  
**Solution:** Install Flask: `pip install flask flask-cors`

**Problem:** Phoropter API not responding  
**Solution:** Check network access and API endpoint in `curl_API.md`

---

## 🎓 Next Steps

1. ✅ Run demo to understand flow
2. ✅ Process existing data
3. ✅ Try interactive session
4. 📖 Read API_USAGE.md for detailed curl examples
5. 🔧 Customize config files for your needs
6. 🚀 Integrate with your application

---

**Ready to start? Run the demo:**
```bash
python -m eye_test_engine.demo_conversation
```
