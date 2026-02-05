# JCC Handle Behavior Clarification

## Current Understanding

Based on the user's clarification: **"when the JCC chart is clicked, it default loads to flip 1 of Axis. no need to call handle or flip APIs."**

## JCC Chart Behavior

### Initial State
When `jcc_chart` is displayed:
- **Automatically shows**: Flip 1 of Axis mode
- **No API call needed**: The chart defaults to this state

### Current Implementation Questions

#### Scenario 1: Transition from Flip 1 to Flip 2 (AUTO_FLIP)
**Question**: When we want to show Flip 2 after the 2-second countdown, do we need to call `jcc_flip("handle")`?

**Current Code** (line 395):
```python
if intent == "AUTO_FLIP":
    self.jcc_flip_state = "flip2"
    self.current_row = self._copy_row_state()
    self._update_state(occluder="Right_Axis_Flip2")
    self.jcc_flip("handle")  # ← Is this needed?
    return self._build_response()
```

**Options**:
- **A**: Keep `jcc_flip("handle")` - physically flips the JCC lens to show Flip 2
- **B**: Remove `jcc_flip("handle")` - Flip 2 is shown automatically somehow?

#### Scenario 2: Reset to Flip 1 after adjustment
**Question**: After patient selects Flip 1 or Flip 2 and we make an adjustment, do we need to call `jcc_flip("handle")` to reset to Flip 1?

**Current Code** (lines 413-415):
```python
self.jcc_flip("increase")  # or "decrease"
self.jcc_flip("handle")     # ← Is this needed to reset to Flip 1?
self.jcc_flip_state = "flip1"
```

**Options**:
- **A**: Keep `jcc_flip("handle")` - resets the JCC lens back to Flip 1
- **B**: Remove `jcc_flip("handle")` - Flip 1 is restored automatically?

#### Scenario 3: Initial JCC Chart Load
**Question**: When transitioning to JCC phase, do we need to call any JCC API?

**Current Code** (lines 734-735):
```python
self.set_chart("jcc_chart")
self.jcc_flip("R")  # ← Is this needed?
```

**Options**:
- **A**: Keep `jcc_flip("R")` - sets the eye mode for JCC operations
- **B**: Remove `jcc_flip("R")` - eye mode is set automatically?

## API Reference

From `curl_API.md`:

### JCC Handle
```bash
curl -X POST .../run-tests \
  -d '{ "test_cases": [{ "jcc": "handle" }] }'
```
**Description**: "Flip the JCC lens handle"

### JCC Eye Modes
```bash
# Test Right Eye
curl -X POST .../run-tests \
  -d '{ "test_cases": [{ "jcc": "R" }] }'
```
**Description**: Sets which eye is being tested

### JCC Increase/Decrease
```bash
curl -X POST .../run-tests \
  -d '{ "test_cases": [{ "jcc": "increase" }] }'
```
**Description**: Adjusts the value in current JCC mode

## Clarification Needed

Please confirm the correct behavior for each scenario:

1. **AUTO_FLIP (Flip 1 → Flip 2)**: Should we call `jcc_flip("handle")`?
   - [ ] Yes - needed to physically flip the lens
   - [ ] No - Flip 2 shows automatically

2. **Reset after adjustment (Flip 2 → Flip 1)**: Should we call `jcc_flip("handle")`?
   - [ ] Yes - needed to reset the lens position
   - [ ] No - Flip 1 restores automatically

3. **Initial JCC load**: Should we call `jcc_flip("R")` or `jcc_flip("L")`?
   - [ ] Yes - needed to set eye mode
   - [ ] No - eye mode is set by occluder only

## Current Implementation Summary

### What we're doing now:
1. ✅ Set JCC chart → defaults to Flip 1 of Axis
2. ✅ Call `jcc_flip("R")` or `jcc_flip("L")` to set eye mode
3. ✅ Wait 2 seconds
4. ✅ Call `jcc_flip("handle")` to show Flip 2
5. ✅ After adjustment, call `jcc_flip("increase")` or `jcc_flip("decrease")`
6. ✅ Call `jcc_flip("handle")` to reset to Flip 1

### What might be redundant:
- Initial `jcc_flip("R")` / `jcc_flip("L")` calls?
- `jcc_flip("handle")` calls during AUTO_FLIP?
- `jcc_flip("handle")` calls after adjustments?

---

**Status**: Awaiting clarification on which JCC API calls are necessary
