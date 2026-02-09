# Test Files Location

## All test files have been moved to:

```
eye_test_engine/tests/
```

## Available Tests

1. **test_prev_state.py** - Tests "Prev State" feature for "Blurry" option
2. **test_prev_state_unable_read.py** - Tests "Prev State" feature for "Unable to read" option
3. **test_no_getting_better.py** - Verifies "Getting better" option is removed
4. **test_jcc_power_zero.py** - Tests JCC Power zero-cylinder logic
5. **demo_prev_state.py** - Interactive demo for "Prev State" feature

## How to Run Tests

```bash
# Navigate to tests folder
cd eye_test_engine/tests

# Run individual test
python test_prev_state.py

# Run all tests
python test_prev_state.py && \
python test_prev_state_unable_read.py && \
python test_no_getting_better.py && \
python test_jcc_power_zero.py
```

## Documentation

See `eye_test_engine/tests/README.md` for detailed information about each test.

## Test Structure

```
eye_test_engine/
├── tests/
│   ├── README.md                      # Test documentation
│   ├── test_prev_state.py            # Prev State for Blurry
│   ├── test_prev_state_unable_read.py # Prev State for Unable to read
│   ├── test_no_getting_better.py     # Getting better removed
│   ├── test_jcc_power_zero.py        # JCC Power zero logic
│   └── demo_prev_state.py            # Interactive demo
├── interactive_session.py             # Main session logic
├── api_server.py                      # Flask API server
└── ...
```

All tests use relative imports and can be run from the tests directory.
