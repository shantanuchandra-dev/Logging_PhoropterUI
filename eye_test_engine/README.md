# Eye Test Engine

Modular end-to-end eye test algorithm for processing phoropter CSV data and annotating with clinical phases.

## Overview

This engine processes curated conversation CSVs from eye tests and:
- Identifies clinical phases (distance vision, refraction, JCC, duochrome, binocular balance)
- Tracks state transitions using a state machine
- Annotates rows with phase information
- Generates summary reports with final prescriptions

## Architecture

```
eye_test_engine/
├── config/
│   ├── protocol.yaml          # Phase definitions and transition rules
│   └── thresholds.yaml         # Configurable thresholds
├── core/
│   ├── context.py              # Row-level context and normalization
│   ├── state_machine.py        # Phase transition logic
│   └── transitions.py          # (future) Advanced transition rules
├── modules/
│   ├── spherical.py            # Sphere refinement logic
│   ├── cylinder_axis.py        # JCC axis refinement
│   ├── cylinder_power.py       # JCC power refinement
│   ├── duochrome.py            # Red/green balance
│   └── binocular_balance.py    # Final verification
├── io/
│   ├── inputs.py               # CSV loading
│   └── outputs.py              # CSV writing and reports
├── analytics/
│   ├── scoring.py              # (future) Quality scoring
│   └── confidence.py           # (future) Confidence analysis
├── tests/
│   └── (test files)
├── run.py                      # Main execution script
└── README.md                   # This file
```

## Installation

```bash
# Install dependencies
pip install pyyaml

# Make run.py executable
chmod +x run.py
```

## Usage

### Process a single CSV file

```bash
python run.py path/to/curated_conversation.csv --output results/ --summary
```

### Process a directory of CSVs

```bash
python run.py Curated_Conversations/ --output results/ --summary --verbose
```

### Command-line options

- `input`: Path to CSV file or directory
- `--output DIR`: Output directory (default: `eye_test_output/`)
- `--summary`: Generate summary reports with final prescriptions
- `--verbose`: Print detailed progress

## Output

### Annotated CSV

Each row includes two additional fields:
- `Phase_ID`: Short phase identifier (A, B, E, F, G, D, H, I, J, K, etc.)
- `Phase_Name`: Human-readable phase name

### Summary Report (with --summary)

Text file containing:
- Total rows processed
- Phase distribution (row counts per phase)
- Final prescription (SPH, CYL, AXIS, ADD for both eyes)
- Start/end timestamps

## Configuration

### protocol.yaml

Defines:
- Phase triggers (occluder states, chart types)
- Questions and intents for each phase
- Exit conditions and transition rules
- Adjustment rules for JCC flips

### thresholds.yaml

Configurable parameters:
- `unable_read_threshold`: Number of "Unable to read" to exit sphere phase (default: 2)
- `axis_increment`: Degrees to adjust per JCC axis flip (default: 5)
- `power_increment`: Diopters to adjust per JCC power flip (default: 0.25)
- `confidence_window_rows`: Rows to look ahead for repetition (default: 3)

## Phase Flow

```
Distance Vision (A)
    ↓
Right Eye Refraction (B) → Pinhole Check (optional)
    ↓ (2x "Unable to read" after SPH)
JCC Axis Right (E)
    ↓ (axis stable)
JCC Power Right (F)
    ↓ (power stable or 0.00)
Duochrome Right (G)
    ↓
Left Eye Refraction (D) → Pinhole Check (optional)
    ↓ (2x "Unable to read" after SPH)
JCC Axis Left (H)
    ↓ (axis stable)
JCC Power Left (I)
    ↓ (power stable or 0.00)
Duochrome Left (J)
    ↓
Binocular Balance (K)
    ↓
Test Complete
```

## Modules

### Spherical Module
- Tracks SPH changes and "Unable to read" count
- Determines when to exit sphere refinement
- Suggests next SPH values

### Cylinder Axis Module
- Analyzes Flip1→Flip2 pairs
- Determines patient choice (GAP/RAM)
- Calculates axis changes
- Detects stability

### Cylinder Power Module
- Analyzes Flip1→Flip2 pairs
- Determines patient choice (GAP/RAM)
- Calculates cylinder changes
- Handles 0.00 exception
- Detects stability

### Duochrome Module
- Parses red/green/both responses
- Provides adjustment recommendations

### Binocular Balance Module
- Analyzes final verification sequence
- Determines if balance is achieved

## State Machine

The `StateMachine` class:
- Tracks current phase and state variables
- Processes each row and determines phase
- Updates counters (unable_read_count, stability flags)
- Triggers phase transitions based on exit conditions

State variables tracked:
- `current_phase`, `current_eye`
- `duochrome_seen` (to distinguish distance vision vs binocular balance)
- `right_sph_stable`, `left_sph_stable`
- `right_axis_stable`, `left_axis_stable`
- `right_cyl_stable`, `left_cyl_stable`
- `unable_read_count_right`, `unable_read_count_left`

## Examples

### Example 1: Process single file with summary

```bash
python run.py Curated_Conversations/8ffQRG3mTI268sHa3N5DVQ.csv --summary
```

Output:
```
✓ Processed 137 rows
  Output: eye_test_output/annotated_8ffQRG3mTI268sHa3N5DVQ.csv
  Summary: eye_test_output/summary_8ffQRG3mTI268sHa3N5DVQ.txt
```

### Example 2: Process directory

```bash
python run.py Curated_Conversations/ --output results/ --verbose
```

Output:
```
Processing 92 files from Curated_Conversations/
  Processing: 0Ut9GmQcRGm1r3F-f0yprg.csv
    ✓ 88 rows
  Processing: 1dumhM-RRQKEggsHCaJoNw.csv
    ✓ 111 rows
  ...

✓ Processed 92 sessions
  Output directory: results/
```

## Testing

(Future: test suite in `tests/` directory)

## Future Enhancements

- `analytics/scoring.py`: Quality scoring for test sessions
- `analytics/confidence.py`: Advanced confidence analysis
- `core/transitions.py`: More sophisticated transition logic
- Real-time execution mode (process rows as they arrive)
- Integration with phoropter hardware
- Machine learning for intent prediction

## References

- `docs/STATE_MACHINE_DIAGRAM.md`: Detailed state machine diagram
- `docs/CURATION_LOGIC.md`: Snellen and JCC curation logic
- `refined_clinical_protocol.txt`: Clinical protocol mapping rules
- `conversation_config.json`: Question and intent templates
