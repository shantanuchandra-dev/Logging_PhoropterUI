# Prev State Feature - Quick Start Guide

## What Was Implemented

Added a "Prev State" button that appears when a user clicks "Blurry" during eye refraction tests. This allows them to undo the power adjustment and return to the previous state.

## How It Works

### User Flow
1. **During Right Eye Test**: "I'm covering your left eye. Please read the line you can see clearly."
2. **User clicks**: "Blurry"
3. **System**: Applies -0.25D SPH adjustment (as before)
4. **NEW**: A "Prev State" button now appears in the options
5. **User clicks**: "Prev State"
6. **System**: Restores the power to what it was before "Blurry" was clicked

### Same for Left Eye
The feature works identically when testing the left eye ("I'm covering your right eye...").

## Files Modified

- `eye_test_engine/interactive_session.py` - Main implementation

## Files Created

- `test_prev_state.py` - Automated test suite (3 test scenarios, all passing ✅)
- `demo_prev_state.py` - Interactive demo to manually test the feature
- `PREV_STATE_FEATURE.md` - Detailed implementation documentation
- `PREV_STATE_VISUAL_GUIDE.md` - Visual guide showing UI flow

## Testing

### Run Automated Tests
```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI
python test_prev_state.py
```

All tests pass ✅:
- ✓ Right eye refraction: Prev State appears after Blurry
- ✓ Right eye refraction: Power restored correctly
- ✓ Left eye refraction: Prev State appears after Blurry
- ✓ Left eye refraction: Power restored correctly
- ✓ Prev State does NOT appear for other intents

### Run Interactive Demo
```bash
python demo_prev_state.py
```

Follow the on-screen prompts to see the feature in action!

## Technical Details

### State Saved Before Blurry
```python
{
    'r_sph': 0.0,      # Right eye sphere
    'r_cyl': 0.0,      # Right eye cylinder
    'r_axis': 180.0,   # Right eye axis
    'l_sph': 0.0,      # Left eye sphere
    'l_cyl': 0.0,      # Left eye cylinder
    'l_axis': 180.0,   # Left eye axis
    'occluder_state': 'Left_Occluded',
    'chart_display': 'snellen_chart_20_20_20'
}
```

### CURL Commands Sent

**When "Blurry" is clicked (Right Eye):**
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"right_eye": {"sph": -0.25}}]}'
```

**When "Prev State" is clicked (Right Eye):**
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"right_eye": {"sph": 0.0}}]}'
```

**When "Blurry" is clicked (Left Eye):**
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"left_eye": {"sph": -0.25}}]}'
```

**When "Prev State" is clicked (Left Eye):**
```bash
curl -X POST https://rajasthan-royals.preprod.lenskart.com/phoropter/phoropter-1/run-tests \
  -H "Content-Type: application/json" \
  -d '{"test_cases": [{"left_eye": {"sph": 0.0}}]}'
```

## API Integration

No changes needed! The existing Flask API (`api_server.py`) and frontend (`app.js`) automatically handle the new "Prev State" intent:

1. Backend adds "Prev State" to the intents list in the JSON response
2. Frontend displays it as a button
3. When clicked, frontend sends "Prev State" to the backend
4. Backend processes it and restores the previous power
5. Frontend updates the UI with the restored values

## Key Features

✅ **Only One Level of Undo**: Prevents confusion by keeping it simple
✅ **Automatic Cleanup**: "Prev State" disappears after use or when another option is selected
✅ **Phase-Specific**: Only appears during refraction phases (right_eye_refraction and left_eye_refraction)
✅ **Full State Restoration**: Restores all power parameters accurately
✅ **No Breaking Changes**: Existing functionality unchanged

## Frontend Experience

The user sees this in the UI:

**Before clicking "Blurry":**
```
Available Options:
  1. Able to read
  2. Blurry
  3. Unable to read
  4. Getting better
```

**After clicking "Blurry":**
```
Available Options:
  1. Able to read
  2. Blurry
  3. Unable to read
  4. Getting better
  5. Prev State        ← NEW!
```

**After clicking "Prev State":**
```
Available Options:
  1. Able to read
  2. Blurry
  3. Unable to read
  4. Getting better
```

## Questions?

See the detailed documentation:
- `PREV_STATE_FEATURE.md` - Full implementation details
- `PREV_STATE_VISUAL_GUIDE.md` - Visual flowcharts and examples

Or run the interactive demo:
```bash
python demo_prev_state.py
```
