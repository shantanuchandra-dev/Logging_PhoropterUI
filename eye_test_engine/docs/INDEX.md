# Eye Test Engine Documentation Index

## Quick Start
- **[QUICK_START.md](QUICK_START.md)** - Get started quickly with the eye test engine

## Core Documentation
- **[README.md](README.md)** - Frontend overview and setup
- **[SUMMARY.md](SUMMARY.md)** - Project summary and architecture
- **[API_USAGE.md](API_USAGE.md)** - API endpoints and usage guide
- **[PHASE_NAMING.md](PHASE_NAMING.md)** - Phase naming convention (A, B, D, E, F, G, H, I, J, K)
- **[PHASE_JUMP_FEATURE.md](PHASE_JUMP_FEATURE.md)** - Phase navigation dropdown in header

## Implementation Details

### Refraction Logic
- **[REFRACTION_LOGIC_UPDATE.md](REFRACTION_LOGIC_UPDATE.md)** - Snellen chart progression and power adjustments

### JCC (Jackson Cross Cylinder) Implementation
- **[JCC_NO_SET_POWER.md](JCC_NO_SET_POWER.md)** - Confirmed: No set_power() during JCC phases (E,F,H,I)
- **[JCC_OPERATIONS_ORDER.md](JCC_OPERATIONS_ORDER.md)** - Correct order: call increase/decrease FIRST, then update state
- **[JCC_EYE_MODE_MAPPING.md](JCC_EYE_MODE_MAPPING.md)** - Correct JCC eye mode mapping (Left_Occluded→L, Right_Occluded→R)
- **[JCC_CHART_BEHAVIOR.md](JCC_CHART_BEHAVIOR.md)** - JCC chart default behavior and API usage
- **[JCC_ADJUSTMENT_STEPS.md](JCC_ADJUSTMENT_STEPS.md)** - JCC adjustment step sizes (±5° axis, ±0.25D power)
- **[JCC_AUTO_FLIP_IMPLEMENTATION.md](JCC_AUTO_FLIP_IMPLEMENTATION.md)** - Auto-flip sequence implementation
- **[AUTO_FLIP_IMPLEMENTATION.md](AUTO_FLIP_IMPLEMENTATION.md)** - Detailed auto-flip mechanics
- **[JCC_BEHAVIOR_VERIFICATION.md](JCC_BEHAVIOR_VERIFICATION.md)** - Verification of JCC phase behaviors
- **[JCC_FLOW_DIAGRAM.md](JCC_FLOW_DIAGRAM.md)** - Visual flow charts and timing diagrams
- **[FLIP2_INTENTS_FIX.md](FLIP2_INTENTS_FIX.md)** - Fix for missing Flip 2 intent buttons

## Testing & Verification
- **[TESTING_CHECKLIST.md](TESTING_CHECKLIST.md)** - Comprehensive testing checklist
- **[FINAL_VERIFICATION_SUMMARY.md](FINAL_VERIFICATION_SUMMARY.md)** - Final verification and status

## Frontend Guides
- **[FRONTEND_GUIDE.md](FRONTEND_GUIDE.md)** - Frontend development guide
- **[DEMO.md](DEMO.md)** - Step-by-step demo walkthrough

## Delivery Summaries
- **[COMPLETE_FRONTEND_DELIVERY.md](COMPLETE_FRONTEND_DELIVERY.md)** - Complete frontend delivery documentation
- **[FINAL_DELIVERY_SUMMARY.md](FINAL_DELIVERY_SUMMARY.md)** - Final delivery summary

---

## Document Organization

### For New Users
1. Start with [QUICK_START.md](QUICK_START.md)
2. Read [README.md](README.md) for frontend setup
3. Follow [DEMO.md](DEMO.md) for a walkthrough

### For Developers
1. Review [SUMMARY.md](SUMMARY.md) for architecture
2. Check [API_USAGE.md](API_USAGE.md) for API details
3. Read implementation docs for specific features

### For Testing
1. Use [TESTING_CHECKLIST.md](TESTING_CHECKLIST.md)
2. Refer to [FINAL_VERIFICATION_SUMMARY.md](FINAL_VERIFICATION_SUMMARY.md)

### For Understanding JCC Logic
1. Start with [JCC_FLOW_DIAGRAM.md](JCC_FLOW_DIAGRAM.md)
2. Read [JCC_BEHAVIOR_VERIFICATION.md](JCC_BEHAVIOR_VERIFICATION.md)
3. Check [FLIP2_INTENTS_FIX.md](FLIP2_INTENTS_FIX.md) for technical details

---

## Latest Updates

### Recent Fixes (Latest)
- **JCC Eye Mode Mapping**: Correct mapping for all phases (Left_Occluded→L, Right_Occluded→R, BINO→BINO)
- **JCC Chart Behavior**: Simplified JCC initialization - chart defaults to Flip 1 of Axis, no extra API calls needed
- **Phase-Specific Logic**: JCC eye mode only set for non-JCC phases (Distance Vision, Refraction, Duochrome, Binocular Balance)
- **Flip 2 Intent Buttons**: Fixed missing intent buttons after AUTO_FLIP
- **Repeat Functionality**: Properly resets to Flip 1 with countdown
- **State Management**: All state changes now use `_update_state()` helper

### Key Features
- ✅ Automatic JCC Flip 1 → Flip 2 transition (2-second countdown)
- ✅ Proper Snellen chart progression (largest to smallest)
- ✅ Intent-based patient response handling
- ✅ Immediate phase transitions (Axis → Power → Duochrome)
- ✅ Full binocular refraction workflow

---

## File Structure

```
eye_test_engine/
├── README.md (main project readme)
├── docs/
│   ├── INDEX.md (this file)
│   ├── QUICK_START.md
│   ├── README.md (frontend readme)
│   ├── SUMMARY.md
│   ├── API_USAGE.md
│   ├── REFRACTION_LOGIC_UPDATE.md
│   ├── JCC_AUTO_FLIP_IMPLEMENTATION.md
│   ├── AUTO_FLIP_IMPLEMENTATION.md
│   ├── JCC_BEHAVIOR_VERIFICATION.md
│   ├── JCC_FLOW_DIAGRAM.md
│   ├── FLIP2_INTENTS_FIX.md
│   ├── TESTING_CHECKLIST.md
│   ├── FINAL_VERIFICATION_SUMMARY.md
│   ├── FRONTEND_GUIDE.md
│   ├── DEMO.md
│   ├── COMPLETE_FRONTEND_DELIVERY.md
│   └── FINAL_DELIVERY_SUMMARY.md
├── config/
├── core/
├── modules/
├── frontend/
└── ...
```
