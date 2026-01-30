import csv
import json
from pathlib import Path


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return json.load(f)


def normalize_chart(value: str) -> str:
    return (value or "").strip().lower()


def is_snellen(chart: str) -> bool:
    return chart.startswith("snellen_chart")


def is_echart(chart: str) -> bool:
    return chart.startswith("echart")


def is_number_chart(chart: str) -> bool:
    return chart.startswith("number_chart")


def get_question(row: dict, config: dict) -> str:
    chart = normalize_chart(row.get("Chart_Display"))
    occ = (row.get("Occluder_State") or "").strip()

    if "Flip1" in occ:
        return config["question_flip1"]
    if "Flip2" in occ:
        return config["question_flip2"]

    if chart == "jcc_chart":
        return config["question_jcc"]

    if chart == "duochrome":
        return config["question_duochrome"]

    if chart == "near_vision" or is_number_chart(chart):
        return config["question_near"]

    if is_snellen(chart):
        if occ == "Left_Occluded":
            return config["question_snellen_left_occluded"]
        if occ == "Right_Occluded":
            return config["question_snellen_right_occluded"]
        return config["question_snellen"]

    if is_echart(chart) or chart == "pictorial_chart":
        return config["question_echart"]

    return config["question_fallback"]


def get_snellen_answer(chart: str) -> str:
    if not chart:
        return "Reads the chart."
    return f"Reads {chart}."


def get_flip_answer(current: dict, nxt: dict) -> str:
    occ = current.get("Occluder_State") or ""
    if not nxt:
        return ""

    if "Axis" in occ:
        key = "R_AXIS" if occ.startswith("Right_") else "L_AXIS"
        try:
            cur = float(current.get(key) or 0)
            nxt_val = float(nxt.get(key) or 0)
        except ValueError:
            return ""
        if nxt_val > cur:
            return "Increase axis."
        if nxt_val < cur:
            return "Decrease axis."
        return "No axis change."

    if "Power" in occ:
        key = "R_CYL" if occ.startswith("Right_") else "L_CYL"
        try:
            cur = float(current.get(key) or 0)
            nxt_val = float(nxt.get(key) or 0)
        except ValueError:
            return ""
        if nxt_val > cur:
            return "Increase cylinder."
        if nxt_val < cur:
            return "Decrease cylinder."
        return "No cylinder change."

    return ""


def get_answer(row: dict, nxt: dict, config: dict) -> str:
    chart = normalize_chart(row.get("Chart_Display"))
    occ = row.get("Occluder_State") or ""

    if "Flip1" in occ:
        return ""
    if "Flip2" in occ:
        return get_flip_answer(row, nxt)

    if is_snellen(chart) and occ == "Left_Occluded":
        return get_snellen_answer(row.get("Chart_Display"))

    if chart == "duochrome":
        return config["answer_duochrome"]

    if chart == "jcc_chart":
        return config["answer_jcc"]

    if chart == "near_vision" or is_number_chart(chart):
        return config["answer_near"]

    if is_echart(chart) or is_snellen(chart) or chart == "pictorial_chart":
        return config["answer_distance"]

    return config["answer_fallback"]


def get_confidence(rows: list[dict], idx: int, question: str, occ: str, config: dict) -> str:
    window = config["confidence_window_rows"]
    for j in range(idx + 1, min(len(rows), idx + 1 + window)):
        nxt = rows[j]
        if get_question(nxt, config) == question and (nxt.get("Occluder_State") or "") == occ:
            return config["confidence_confused_label"]
    return config["confidence_confident_label"]


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "conversation_config.json")

    source_dir = root / config["input_valid_dir"]
    target_dir = root / config["output_curated_dir"]
    target_dir.mkdir(exist_ok=True)

    move_files = config["move_files"]
    add_fields = config["output_fields"]

    for csv_path in sorted(source_dir.glob("*.csv")):
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                continue

        new_fieldnames = reader.fieldnames + add_fields
        updated_rows = []
        for idx, row in enumerate(rows):
            nxt = rows[idx + 1] if idx + 1 < len(rows) else None
            question = get_question(row, config)
            answer = get_answer(row, nxt, config)
            confidence = get_confidence(rows, idx, question, row.get("Occluder_State") or "", config)

            new_row = dict(row)
            new_row[add_fields[0]] = question
            new_row[add_fields[1]] = answer
            new_row[add_fields[2]] = confidence
            updated_rows.append(new_row)

        out_path = target_dir / csv_path.name
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_fieldnames)
            writer.writeheader()
            writer.writerows(updated_rows)

        if move_files:
            csv_path.unlink()

    print(f"Processed {len(list(target_dir.glob('*.csv')))} files to {target_dir}")


if __name__ == "__main__":
    main()
