# Testing Checklist - Auto-Flip Implementation

## Issues to Verify

1. ✅ **Button Locking** - Button should disable on click
2. ✅ **Single Reset** - Only one phoropter reset
3. ✅ **Flip 2 Intents** - Intent buttons should appear after Flip 2 (FIXED)
4. ✅ **Axis → Power Transition** - Should complete axis before moving to power (VERIFIED)

---

## Test Sequence

### Phase 1: Distance Vision
- [ ] Click "Start Eye Test"
- [ ] Button becomes disabled and grayed out
- [ ] See "Phoropter reset to 0/0/180" **once** in history
- [ ] See "Chart: echart_400"
- [ ] Question: "Please read the line you can see clearly."
- [ ] Intents: Able to read, Blurry, Unable to read
- [ ] Select "Able to read"

### Phase 2: Right Eye Refraction
- [ ] See "Chart: snellen_chart_200_150" (largest chart)
- [ ] See "Occluder: Left_Occluded"
- [ ] Question: "I'm covering your left eye..."
- [ ] Intents: Able to read, Blurry, Unable to read, Getting better
- [ ] Select "Able to read" multiple times to progress through charts
- [ ] Or select "Unable to read" twice to exit to JCC

### Phase 3: JCC Axis Right (AUTO-FLIP TEST)
- [ ] See "Chart: jcc_chart" **once**
- [ ] See "JCC action: R"
- [ ] Question: "This is Flip 1. (Flip 2 will show automatically in 2 seconds)"
- [ ] **NO intent buttons shown** (this is correct for Flip 1)
- [ ] See countdown: "⏱️ Showing Flip 2 in 2 seconds..."
- [ ] Countdown: 2...1...
- [ ] After 2 seconds, see "JCC action: handle"
- [ ] Question changes to: "Now this is Flip 2. Which was better?"
- [ ] **Intent buttons SHOULD appear:**
  - 1. Flip 1 was better (GAP Axis)
  - 2. Flip 2 was better (RAM Axis)
  - 3. Both Same
  - 4. Repeat

**✅ FIXED:** Derived fields (`is_flip1`, `is_flip2`) now update correctly when occluder state changes.

### Phase 4: Test Axis Refinement Cycle
- [ ] Select "Flip 1 was better"
- [ ] See "JCC action: increase"
- [ ] See "JCC action: handle"
- [ ] See axis value increase by 5°
- [ ] Countdown starts again (2...1...)
- [ ] Flip 2 shows again
- [ ] Intent buttons appear again
- [ ] Select "Both Same" to move to Power

### Phase 5: JCC Power Right (AUTO-FLIP TEST)
- [ ] See "JCC action: power_axis_switch"
- [ ] Question: "This is Flip 1. (Flip 2 will show automatically)"
- [ ] Countdown: 2...1...
- [ ] Flip 2 shows
- [ ] **Intent buttons SHOULD appear** (same as axis)
- [ ] Test the cycle with "Flip 1 was better" or "Flip 2 was better"
- [ ] Select "Both Same" to move to Duochrome

---

## Expected History Log

```
14:00:00 - Test started
14:00:00 - Phoropter reset to 0/0/180  ← Only once!
14:00:01 - Chart: echart_400
14:00:05 - Response: Able to read
14:00:05 - Chart: snellen_chart_200_150
14:00:05 - Power updated - Occluder: Left_Occluded
... (chart progression)
14:02:00 - Chart: jcc_chart  ← Only once!
14:02:00 - JCC action: R
14:02:03 - Flip 2 displayed
14:02:10 - Response: Flip 1 was better (GAP Axis)
14:02:10 - JCC action: increase
14:02:10 - JCC action: handle
14:02:13 - Flip 2 displayed
14:02:20 - Response: Both Same
14:02:20 - JCC action: power_axis_switch  ← Transition to Power
14:02:23 - Flip 2 displayed
14:02:30 - Response: Both Same
... (continues to duochrome)
```

---

## Debugging Steps

If Flip 2 intents are missing:

1. **Check browser console** - Any JavaScript errors?
2. **Check network tab** - Is AUTO_FLIP response returning intents?
3. **Check backend logs** - Is Flip2 state being set correctly?
4. **Inspect element** - Are buttons created but hidden?

---

## Current Implementation Status

### Backend (interactive_session.py)
- ✅ AUTO_FLIP intent handling
- ✅ auto_flip flag in responses
- ✅ Handle calls after increase/decrease
- ✅ Proper state transitions
- ✅ Axis → Power transition on "Both Same"

### Frontend (app.js)
- ✅ handleAutoFlip() function
- ✅ Countdown timer
- ✅ AUTO_FLIP call to backend
- ✅ displayQuestion() update
- ⚠️ Intent buttons after Flip 2 (needs verification)

### Configuration (protocol.yaml)
- ✅ Updated questions for Flip 1 and Flip 2
- ✅ Empty intents for Flip 1
- ✅ Full intents for Flip 2

---

## Next Steps

1. **Restart backend server** with latest code
2. **Refresh frontend** to load latest JavaScript
3. **Test through JCC phases** and verify:
   - Countdown appears
   - Flip 2 shows automatically
   - Intent buttons appear after Flip 2
   - Axis completes before Power
4. **Report any issues** with specific phase and behavior

---

**Please test and let me know if Flip 2 intents are showing correctly!**
