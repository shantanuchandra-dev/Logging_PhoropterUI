# Phoropter UI

Standalone web-based eye test dashboard for remote optometrists.

## Quick Start

```bash
cd phoropter-ui
python3 -m http.server 8080
```

Open **http://localhost:8080** in your browser.

## Features

- **AR / Lensometer** data entry with save/load (localStorage)
- **Subjective Refraction** controls with ±0.25 SPH/CYL/ADD, ±1/±5 AXIS
- **3 Memory Slots** for trial value snapshots
- **Live Video** via URL or webcam
- **CSV / JSON export** with patient + branch info
- **Quick Compare** table (AR vs Lenso vs Subjective)
- **Session Log** with timestamped entries

## Keyboard Workflow

1. Fill patient/branch info in the header
2. Enter AR and Lensometer values (left panel)
3. Use subjective refraction controls (center) to adjust R/L/Both
4. Save trial values to memory slots (right panel)
5. Click **Finalize Prescription** → Export CSV or JSON
