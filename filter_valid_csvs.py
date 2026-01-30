import csv
import json
from pathlib import Path


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return json.load(f)


def normalize_chart(value: str) -> str:
    return (value or "").strip().lower()


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "conversation_config.json")

    source_dir = root / config["input_analyzed_dir"]
    target_dir = root / config["output_valid_dir"]
    target_dir.mkdir(exist_ok=True)

    required_chart_types = set(config["required_chart_types"])
    required_occluders = {o.upper() for o in config["required_occluder_states"]}

    restart_state_values = config["restart_state_row"]
    copied = []
    skipped_no_restart = []
    skipped_requirements = []

    for csv_path in sorted(source_dir.glob("*.csv")):
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                continue

        restart_index = None
        for i, row in enumerate(rows):
            match = True
            for key, val in restart_state_values.items():
                if row.get(key, "") != val:
                    match = False
                    break
            if match:
                restart_index = i
                break

        if restart_index is None:
            skipped_no_restart.append(csv_path.name)
            continue

        trimmed_rows = rows[restart_index:]

        chart_types = set()
        occluders = set()
        for row in trimmed_rows:
            chart_display = normalize_chart(row.get("Chart_Display"))
            for chart_type in required_chart_types:
                if chart_type in chart_display:
                    chart_types.add(chart_type)
            occ = (row.get("Occluder_State") or "").upper()
            if occ:
                occluders.add(occ)

        if not required_chart_types.issubset(chart_types) or not required_occluders.issubset(
            occluders
        ):
            skipped_requirements.append(csv_path.name)
            continue

        out_path = target_dir / csv_path.name
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(trimmed_rows)

        copied.append(csv_path.name)

    print(f"Copied: {len(copied)}")
    print(f"Skipped (no restart row): {len(skipped_no_restart)}")
    print(f"Skipped (requirements): {len(skipped_requirements)}")
    if copied:
        print("Sample copied files:")
        for name in copied[:10]:
            print("  ", name)


if __name__ == "__main__":
    main()
