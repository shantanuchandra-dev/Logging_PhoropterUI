# Processing UI Update - Hide Intents During Processing

## Summary

Updated the frontend to hide intent buttons while processing a response, showing only a "Processing..." message until the backend completes and the phoropter is updated.

## Changes Made

### File: `eye_test_engine/frontend/app.js`

#### 1. Updated `submitIntent()` Function

**Before:**
- Disabled intent buttons but kept them visible
- Displayed new intents immediately after receiving backend response
- Phoropter update happened after displaying intents

**After:**
- Replaces all intent buttons with "Processing..." message immediately when clicked
- Updates phoropter first
- Displays new intents only AFTER all processing is complete

**Key Changes:**
```javascript
// Hide all intent buttons during processing
const intentButtonsContainer = document.getElementById('intentButtons');
intentButtonsContainer.innerHTML = '<div class="alert alert-info">Processing...</div>';

// ... backend call ...

// Update phoropter first
await setPhoropter(data);

// Display question and intents AFTER processing is complete
displayQuestion(data);
```

#### 2. Updated `handleAutoFlip()` Function

**Before:**
- Disabled intent buttons but kept them visible during countdown

**After:**
- Completely hides intent buttons during auto-flip countdown
- Shows countdown timer in question box
- Displays intents only after Flip 2 is shown

**Key Changes:**
```javascript
// Hide intent buttons during auto-flip countdown
const intentButtonsContainer = document.getElementById('intentButtons');
intentButtonsContainer.innerHTML = '';

// ... countdown and auto-flip ...

// Display intents after flip is complete
displayQuestion(data);
```

## User Experience

### Before
1. User clicks an intent button (e.g., "Blurry")
2. Buttons become disabled (grayed out) but remain visible
3. New intents appear while phoropter is still updating
4. User might see old intents briefly before new ones load

### After
1. User clicks an intent button (e.g., "Blurry")
2. **All intent buttons disappear immediately**
3. **"Processing..." message is shown**
4. Phoropter updates
5. **New intents appear only when everything is ready**

### During Auto-Flip (JCC Phases)
1. Flip 1 is shown
2. **Intent buttons are hidden**
3. Countdown timer shows: "⏱️ Showing Flip 2 in X seconds..."
4. Flip 2 is shown
5. **Intent buttons appear with options**

## Benefits

✅ **Clearer UI State**: User knows when system is processing
✅ **Prevents Double-Clicks**: No visible buttons to click during processing
✅ **Better UX**: No flickering between old and new intents
✅ **Consistent Behavior**: Same pattern for all intent submissions
✅ **Professional Look**: Clean "Processing..." message instead of disabled buttons

## Technical Details

### Processing Flow
```
User clicks intent
    ↓
Hide all intents, show "Processing..."
    ↓
Send request to backend
    ↓
Receive response
    ↓
Update session info
    ↓
Update phoropter (CURL commands)
    ↓
Display new question and intents
    ↓
User can interact again
```

### Auto-Flip Flow
```
Backend requests auto-flip
    ↓
Hide all intents
    ↓
Show countdown timer
    ↓
Wait X seconds
    ↓
Send AUTO_FLIP to backend
    ↓
Update phoropter
    ↓
Display Flip 2 question and intents
    ↓
User can respond
```

## Testing

To test this change:

1. **Start a test session**
2. **Click any intent** (e.g., "Able to read")
3. **Observe**: Intent buttons should disappear and "Processing..." should appear
4. **Wait**: Buttons reappear with new options after processing completes
5. **During JCC phases**: Buttons should be hidden during countdown
6. **After Flip 2**: Buttons should appear with response options

## No Backend Changes Required

This is purely a frontend UI improvement. The backend API remains unchanged.

## Compatibility

- Works with all existing phases (refraction, JCC, duochrome, etc.)
- Compatible with "Prev State" feature
- Compatible with auto-flip functionality
- No breaking changes to existing functionality
