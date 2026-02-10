# Power Control Redesign

## Overview

The power control system has been redesigned to separate power entry from power application. Users can now store AR and Lenso power values, then choose which one to apply using dedicated buttons outside the modals.

## New Workflow

### 1. Store Power Values

**Set AR Power**:
1. Click "Set AR Power" button
2. Modal opens
3. Enter complete values for both eyes (SPH, CYL, AXIS)
4. Click "Save"
5. Values are stored in memory
6. "AR" apply button becomes enabled

**Set Lenso Power**:
1. Click "Set Lenso Power" button
2. Modal opens
3. Enter complete values for both eyes (SPH, CYL, AXIS)
4. Click "Save"
5. Values are stored in memory
6. "Lenso" apply button becomes enabled

### 2. Apply Power

After storing values, use the "Apply" buttons:

**Options**:
- **AR** - Apply stored AR power (enabled only if AR values are saved)
- **Lenso** - Apply stored Lenso power (enabled only if Lenso values are saved)
- **None** - Clear power (set to 0, always enabled)

## UI Layout

```
┌──────────────────────────────────────────────────────────────┐
│ Eye Test Engine                                              │
│                                                              │
│ [Set AR Power] [Set Lenso Power] │ Apply: [AR] [Lenso] [None]│
└──────────────────────────────────────────────────────────────┘
```

### Button States

#### Initial State (No Power Stored)
```
Apply: [AR (disabled)] [Lenso (disabled)] [None (active)]
```

#### After Storing AR Power
```
Apply: [AR (enabled)] [Lenso (disabled)] [None (active)]
```

#### After Storing Both Powers
```
Apply: [AR (enabled)] [Lenso (enabled)] [None (active)]
```

#### After Applying AR Power
```
Apply: [AR (active, enabled)] [Lenso (enabled)] [None]
```

## Features

### 1. Validation

**Complete Data Required**:
- All 6 values must be entered (SPH, CYL, AXIS for both eyes)
- If any value is missing, alert is shown: "Please enter complete power values for both eyes"
- Save button only works when all fields are filled

**Session Required**:
- Apply buttons require active session
- If no session, alert: "Please start a test session first."

### 2. Button Enabling Logic

**AR Button**:
- Disabled by default
- Enabled after AR power is saved
- Tooltip when disabled: "Set AR Power first"
- Tooltip when enabled: "Apply AR Power"

**Lenso Button**:
- Disabled by default
- Enabled after Lenso power is saved
- Tooltip when disabled: "Set Lenso Power first"
- Tooltip when enabled: "Apply Lenso Power"

**None Button**:
- Always enabled
- Used to clear power (set to 0)

### 3. Visual Feedback

**Active State**:
- Selected button has gradient background (purple-blue)
- White text
- Clear visual indicator of current selection

**Disabled State**:
- Grayed out (40% opacity)
- Not clickable
- Cursor shows "not-allowed"

**Hover State**:
- Light blue background
- Slight elevation (translateY -1px)
- Border color changes to purple-blue

### 4. Power Storage

**In-Memory Storage**:
```javascript
storedPower = {
    ar: {
        right: { sph: -1.25, cyl: -0.50, axis: 180 },
        left: { sph: -1.00, cyl: -0.75, axis: 90 }
    },
    lenso: {
        right: { sph: -1.50, cyl: -0.25, axis: 180 },
        left: { sph: -1.25, cyl: -0.50, axis: 90 }
    }
}
```

**Current Applied Power**:
```javascript
currentAppliedPower = 'none'  // or 'ar' or 'lenso'
```

## Technical Implementation

### HTML Structure

#### Header with Apply Buttons
```html
<div style="display: flex; gap: 15px; align-items: center;">
    <button class="ar-power-btn" onclick="openArPowerModal()">Set AR Power</button>
    <button class="ar-power-btn" onclick="openLensoPowerModal()">Set Lenso Power</button>
    
    <div style="border-left: 2px solid #ddd; padding-left: 15px; display: flex; gap: 10px; align-items: center;">
        <span style="color: #666; font-weight: 600; font-size: 0.9em;">Apply:</span>
        <button id="applyArBtn" class="power-apply-btn" onclick="applyStoredPower('ar')" disabled>
            AR
        </button>
        <button id="applyLensoBtn" class="power-apply-btn" onclick="applyStoredPower('lenso')" disabled>
            Lenso
        </button>
        <button class="power-apply-btn power-apply-none" onclick="applyStoredPower('none')">
            None
        </button>
    </div>
</div>
```

#### Simplified Modals (No Eye Selection)
```html
<div class="modal-backdrop" id="arPowerModal">
    <div class="modal">
        <h3>Set AR Power</h3>
        <p>Enter autorefractor power values. Click Save to store them.</p>
        
        <div class="modal-row">
            <label>Right</label>
            <div class="modal-grid">
                <input id="arRightSph" type="number" step="0.25" placeholder="SPH">
                <input id="arRightCyl" type="number" step="0.25" placeholder="CYL">
                <input id="arRightAxis" type="number" step="1" min="0" max="180" placeholder="AXIS">
            </div>
        </div>
        
        <div class="modal-row">
            <label>Left</label>
            <div class="modal-grid">
                <input id="arLeftSph" type="number" step="0.25" placeholder="SPH">
                <input id="arLeftCyl" type="number" step="0.25" placeholder="CYL">
                <input id="arLeftAxis" type="number" step="1" min="0" max="180" placeholder="AXIS">
            </div>
        </div>
        
        <div class="modal-actions">
            <button class="btn btn-secondary" onclick="closeArPowerModal()">Cancel</button>
            <button class="btn btn-primary" onclick="saveArPower()">Save</button>
        </div>
    </div>
</div>
```

### CSS Styles

```css
.power-apply-btn {
    padding: 6px 16px;
    background: #f5f5f5;
    color: #333;
    border: 2px solid #ddd;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    font-size: 0.9em;
    transition: all 0.2s;
}

.power-apply-btn:hover:not(:disabled) {
    background: #e3f2fd;
    border-color: #667eea;
    transform: translateY(-1px);
}

.power-apply-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

.power-apply-btn.active {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-color: #667eea;
}

.power-apply-none {
    background: #fff3e0;
    border-color: #ff9800;
    color: #f57c00;
}

.power-apply-none.active {
    background: #ff9800;
    color: white;
    border-color: #f57c00;
}
```

### JavaScript Functions

#### Save Functions
```javascript
function saveArPower() {
    // Parse all values
    const rightSph = parseArValue(document.getElementById('arRightSph').value, null);
    const rightCyl = parseArValue(document.getElementById('arRightCyl').value, null);
    const rightAxis = parseArValue(document.getElementById('arRightAxis').value, null);
    const leftSph = parseArValue(document.getElementById('arLeftSph').value, null);
    const leftCyl = parseArValue(document.getElementById('arLeftCyl').value, null);
    const leftAxis = parseArValue(document.getElementById('arLeftAxis').value, null);

    // Validate completeness
    const rightComplete = rightSph !== null && rightCyl !== null && rightAxis !== null;
    const leftComplete = leftSph !== null && leftCyl !== null && leftAxis !== null;

    if (!rightComplete || !leftComplete) {
        alert('Please enter complete power values for both eyes (SPH, CYL, AXIS).');
        return;
    }

    // Store values
    storedPower.ar = {
        right: { sph: rightSph, cyl: rightCyl, axis: rightAxis },
        left: { sph: leftSph, cyl: leftCyl, axis: leftAxis }
    };

    // Enable button
    document.getElementById('applyArBtn').disabled = false;
    document.getElementById('applyArBtn').title = 'Apply AR Power';

    addToHistory('AR power values saved', 'info');
    closeArPowerModal();
}
```

#### Apply Function
```javascript
async function applyStoredPower(type) {
    if (!sessionState.sessionId) {
        alert('Please start a test session first.');
        return;
    }

    // Update button states
    updatePowerButtonStates(type);

    if (type === 'none') {
        // Clear power
        const power = {
            right: { sph: 0, cyl: 0, axis: 180 },
            left: { sph: 0, cyl: 0, axis: 180 }
        };
        await setPower(power, 'BINO');
        currentAppliedPower = 'none';
        addToHistory('Power cleared (None)', 'info');
        return;
    }

    // Apply AR or Lenso power
    const power = type === 'ar' ? storedPower.ar : storedPower.lenso;
    
    if (!power) {
        alert(`No ${type.toUpperCase()} power values stored. Please set them first.`);
        return;
    }

    await setPower(power, 'BINO');
    currentAppliedPower = type;
    
    const label = type === 'ar' ? 'AR' : 'Lenso';
    addToHistory(`${label} power applied`, 'info');
    
    // Update frontend display
    document.getElementById('rightPower').textContent =
        `${power.right.sph.toFixed(2)} / ${power.right.cyl.toFixed(2)} / ${power.right.axis.toFixed(0)}°`;
    document.getElementById('leftPower').textContent =
        `${power.left.sph.toFixed(2)} / ${power.left.cyl.toFixed(2)} / ${power.left.axis.toFixed(0)}°`;
}
```

#### Button State Update
```javascript
function updatePowerButtonStates(activeType) {
    const arBtn = document.getElementById('applyArBtn');
    const lensoBtn = document.getElementById('applyLensoBtn');
    const noneBtn = document.querySelector('.power-apply-none');

    // Remove active class from all
    arBtn.classList.remove('active');
    lensoBtn.classList.remove('active');
    noneBtn.classList.remove('active');

    // Add active class to selected
    if (activeType === 'ar') {
        arBtn.classList.add('active');
    } else if (activeType === 'lenso') {
        lensoBtn.classList.add('active');
    } else if (activeType === 'none') {
        noneBtn.classList.add('active');
    }
}
```

## User Workflows

### Workflow 1: Apply AR Power

```
1. Click "Set AR Power"
   ↓
2. Modal opens
   ↓
3. Enter values:
   Right: SPH -1.25, CYL -0.50, AXIS 180
   Left: SPH -1.00, CYL -0.75, AXIS 90
   ↓
4. Click "Save"
   ↓
5. Modal closes
   History: "AR power values saved"
   AR button becomes enabled
   ↓
6. Click "AR" apply button
   ↓
7. CURL command sent:
   {"test_cases": [{
     "right_eye": {"sph": -1.25, "cyl": -0.50, "axis": 180},
     "left_eye": {"sph": -1.00, "cyl": -0.75, "axis": 90}
   }]}
   ↓
8. Frontend updates:
   Right Power: -1.25 / -0.50 / 180°
   Left Power: -1.00 / -0.75 / 90°
   History: "AR power applied"
   AR button becomes active (highlighted)
```

### Workflow 2: Switch Between AR and Lenso

```
1. AR power already stored and applied (AR button active)
   ↓
2. Click "Set Lenso Power"
   ↓
3. Enter Lenso values and Save
   Lenso button becomes enabled
   ↓
4. Click "Lenso" apply button
   ↓
5. Lenso power applied to phoropter
   AR button becomes inactive
   Lenso button becomes active
   Frontend displays Lenso power values
   ↓
6. Click "AR" apply button again
   ↓
7. AR power re-applied to phoropter
   Lenso button becomes inactive
   AR button becomes active
   Frontend displays AR power values
```

### Workflow 3: Clear Power

```
1. AR or Lenso power currently applied
   ↓
2. Click "None" button
   ↓
3. CURL command sent:
   {"test_cases": [{
     "right_eye": {"sph": 0, "cyl": 0, "axis": 180},
     "left_eye": {"sph": 0, "cyl": 0, "axis": 180}
   }]}
   ↓
4. Frontend updates:
   Right Power: 0.00 / 0.00 / 180°
   Left Power: 0.00 / 0.00 / 180°
   History: "Power cleared (None)"
   None button becomes active
```

## Benefits

### 1. Separation of Concerns
- **Store** values separately from **applying** them
- Can switch between AR and Lenso without re-entering values
- Clear workflow: Set → Save → Apply

### 2. Flexibility
- Store both AR and Lenso values
- Switch between them with one click
- No need to re-enter values

### 3. Validation
- Ensures complete data before enabling apply buttons
- Clear feedback when data is missing
- Prevents partial power application

### 4. User Experience
- Simpler modals (no radio buttons)
- Clear visual feedback (active button highlighted)
- Intuitive workflow

### 5. Safety
- "None" option to clear power
- Always enabled as a safety fallback
- Clear indication of current applied power

## Error Handling

### Missing Values
```
User clicks "Save" with incomplete data
→ Alert: "Please enter complete power values for both eyes (SPH, CYL, AXIS)."
→ Modal stays open
→ Button remains disabled
```

### No Session
```
User clicks apply button without starting test
→ Alert: "Please start a test session first."
→ No CURL command sent
```

### No Stored Values
```
User somehow clicks enabled button but no values stored (edge case)
→ Alert: "No AR power values stored. Please set them first."
→ No CURL command sent
```

## Testing Checklist

### AR Power
- [ ] Click "Set AR Power" opens modal
- [ ] Modal shows "Set AR Power" title
- [ ] Modal has "Save" button (not "Go")
- [ ] Entering incomplete values shows alert
- [ ] Entering complete values enables AR apply button
- [ ] AR button shows "Apply AR Power" tooltip when enabled
- [ ] Clicking AR apply button sends correct CURL
- [ ] Frontend displays AR power values
- [ ] AR button becomes active (highlighted)
- [ ] History shows "AR power values saved" then "AR power applied"

### Lenso Power
- [ ] Click "Set Lenso Power" opens modal
- [ ] Modal shows "Set Lenso Power" title
- [ ] Modal has "Save" button (not "Go")
- [ ] Entering incomplete values shows alert
- [ ] Entering complete values enables Lenso apply button
- [ ] Lenso button shows "Apply Lenso Power" tooltip when enabled
- [ ] Clicking Lenso apply button sends correct CURL
- [ ] Frontend displays Lenso power values
- [ ] Lenso button becomes active (highlighted)
- [ ] History shows "Lenso power values saved" then "Lenso power applied"

### None Button
- [ ] None button is enabled by default
- [ ] None button is active by default (highlighted)
- [ ] Clicking None clears power (sets to 0)
- [ ] Frontend displays 0.00 / 0.00 / 180°
- [ ] History shows "Power cleared (None)"

### Switching
- [ ] Can switch from AR to Lenso
- [ ] Can switch from Lenso to AR
- [ ] Can switch from AR/Lenso to None
- [ ] Active button always highlighted
- [ ] Only one button active at a time

### Button States
- [ ] AR button disabled initially
- [ ] Lenso button disabled initially
- [ ] None button enabled initially
- [ ] Disabled buttons show "not-allowed" cursor
- [ ] Disabled buttons have 40% opacity
- [ ] Enabled buttons respond to hover
- [ ] Active buttons have gradient background

## Files Modified

1. **`eye_test_engine/frontend/index.html`**
   - Added Apply buttons next to Set buttons
   - Removed "Apply to" radio buttons from modals
   - Changed modal button from "Go" to "Save"
   - Updated modal titles and descriptions
   - Added CSS for power apply buttons

2. **`eye_test_engine/frontend/app.js`**
   - Added `storedPower` object for storing values
   - Added `currentAppliedPower` variable
   - Replaced `applyArPower()` with `saveArPower()`
   - Replaced `applyLensoPower()` with `saveLensoPower()`
   - Added `applyStoredPower(type)` function
   - Added `updatePowerButtonStates(activeType)` function
   - Updated `parseArValue()` to handle null properly
   - Initialize "None" button as active on page load

## Conclusion

This redesign provides a cleaner, more intuitive workflow for managing power values. Users can store multiple power sources and switch between them easily, with clear visual feedback and validation at every step.
