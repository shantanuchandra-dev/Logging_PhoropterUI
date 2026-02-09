# Eye Test Engine - Tests

This folder contains all test scripts and demos for the Eye Test Engine.

## Test Files

### 1. `test_prev_state.py`
Tests the "Prev State" functionality for the "Blurry" option.

**What it tests:**
- ✓ Right eye refraction: "Prev State" appears after "Blurry"
- ✓ Right eye refraction: Power restored correctly
- ✓ Left eye refraction: "Prev State" appears after "Blurry"
- ✓ Left eye refraction: Power restored correctly
- ✓ "Prev State" does NOT appear for other intents

**Run:**
```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests
python test_prev_state.py
```

### 2. `test_prev_state_unable_read.py`
Tests the "Prev State" functionality for the "Unable to read" option.

**What it tests:**
- ✓ "Prev State" appears after "Unable to read"
- ✓ "Prev State" still appears after "Blurry"
- ✓ "Prev State" does NOT appear after "Able to read"
- ✓ Power is correctly restored when "Prev State" is clicked

**Run:**
```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests
python test_prev_state_unable_read.py
```

### 3. `test_no_getting_better.py`
Verifies that the "Getting better" option has been successfully commented out.

**What it tests:**
- ✓ Right eye refraction: "Getting better" not in intents list
- ✓ Left eye refraction: "Getting better" not in intents list
- ✓ No linter errors

**Run:**
```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests
python test_no_getting_better.py
```

### 4. `test_jcc_power_zero.py`
Tests JCC Power refinement logic when cylinder is 0.0 and patient says "Flip 1 is better".

**What it tests:**
- ✓ Right eye: First time repeats flip cycle
- ✓ Right eye: Second time moves to duochrome
- ✓ Left eye: First time repeats flip cycle
- ✓ Left eye: Second time moves to duochrome

**Run:**
```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests
python test_jcc_power_zero.py
```

## Demo Files

### `demo_prev_state.py`
Interactive demo to manually test the "Prev State" feature.

**Features:**
- Shows exactly when "Prev State" appears
- Allows manual interaction with the test flow
- Displays current state and available options
- Highlights "Prev State" option with ⭐ star

**Run:**
```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests
python demo_prev_state.py
```

## Running All Tests

To run all tests at once:

```bash
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests

# Run all tests
python test_prev_state.py && \
python test_prev_state_unable_read.py && \
python test_no_getting_better.py && \
python test_jcc_power_zero.py

echo "All tests completed!"
```

## Test Results Summary

| Test | Status | Notes |
|------|--------|-------|
| test_prev_state.py | ✅ Passing | All scenarios tested |
| test_prev_state_unable_read.py | ⚠️ Ready | Needs to be run |
| test_no_getting_better.py | ✅ Passing | Verified removal |
| test_jcc_power_zero.py | ✅ Passing | Main tests passing |

## Notes

- All test files use relative imports from the parent directory
- Tests can be run from anywhere as long as the path is correct
- Demo files are interactive and require user input
- All tests print detailed output showing what's being tested

## Troubleshooting

If you get import errors:
```bash
# Make sure you're in the tests directory
cd /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests

# Or use absolute path
python /Users/shantanuchandra/Downloads/Logging_PhoropterUI/eye_test_engine/tests/test_prev_state.py
```

If tests fail:
1. Check that the backend is not running (tests use the session directly)
2. Verify all dependencies are installed
3. Check that `interactive_session.py` is in the parent directory
4. Review the test output for specific error messages
