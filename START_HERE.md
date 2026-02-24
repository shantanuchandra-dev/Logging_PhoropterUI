# START HERE — Eye Test Engine

## What This Is

A complete web application that:
1. Asks questions one by one through all 10 phases
2. Shows intent options as clickable buttons
3. Automatically controls the phoropter via API calls
4. Displays real-time power, phase, and history

---

## Quick Start (30 Seconds)

```bash
# Option 1: One-Click
cd eye_test_engine && ./start_frontend.sh

# Option 2: Manual
python -m eye_test_engine.api_server          # Terminal 1
cd eye_test_engine/frontend && python3 -m http.server 8080  # Terminal 2
open http://localhost:8080
```

---

## 10 Test Phases

| Phase | Name | Setup |
|-------|------|-------|
| A | Distance Vision | BINO + E-chart |
| B | Right Eye Refraction | Left_Occluded + Snellen |
| E | JCC Axis Right | JCC chart, Flip1/2 |
| F | JCC Power Right | JCC chart, Flip1/2 |
| G | Duochrome Right | Left_Occluded + Duochrome |
| D | Left Eye Refraction | Right_Occluded + Snellen |
| H | JCC Axis Left | JCC chart, Flip1/2 |
| I | JCC Power Left | JCC chart, Flip1/2 |
| J | Duochrome Left | Right_Occluded + Duochrome |
| K | Binocular Balance | BINO + chart_20 |

---

## Documentation

All docs are in `docs/`:

| File | Contents |
|------|---------|
| `docs/API_REFERENCE.md` | Phoropter curl commands + Eye Test Engine API |
| `docs/JCC_FEATURE.md` | JCC auto-flip, operations, adjustments, fixes |
| `docs/PREV_STATE_FEATURE.md` | One-level undo for refraction phases |
| `docs/CHART_SELECTOR.md` | In-session chart switching (Phase A & B) |
| `docs/BINO_FEATURE.md` | Binocular balance phase (Phase K) |
| `docs/PHASE_MANAGEMENT.md` | Phase naming, jump-to-phase feature |
| `docs/FRONTEND_GUIDE.md` | UI layout, power controls, debugging |
| `docs/TESTING.md` | Test files, checklists, expected output |
| `docs/REFRACTION_LOGIC.md` | State machine, chart logic, curation, optotypes |

Standalone docs (kept as-is):
- `leadership_proposal.md` — Executive proposal
- `topcon-roi-spec.md` — UI image analysis ROI spec
- `phoropter-ui/README.md` — Standalone phoropter dashboard

---

## Requirements

```bash
pip install flask flask-cors pyyaml
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| "Failed to start test" | Run `python -m eye_test_engine.api_server` |
| "Could not update phoropter" | Check phoropter URL/ID in frontend header |
| CORS errors | Use `python3 -m http.server 8080`, not file:// |
