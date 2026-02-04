import csv
import json
from pathlib import Path


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return json.load(f)


def normalize_chart(value: str) -> str:
    return (value or "").strip().lower()


def metric_to_imperial(metric_value: float) -> float:
    """Convert metric Snellen value (6/x) to imperial (20/x)."""
    conversion_map = {
        60: 200,
        30: 100,
        20: 70,
        15: 50,
        12: 40,
        9: 30,
        7.5: 25,
        6: 20,
        5: 16,
        4: 13,
        3: 10,
    }
    # Direct match in conversion map
    if metric_value in conversion_map:
        return float(conversion_map[metric_value])
    # Find nearest match
    nearest_key = min(conversion_map.keys(), key=lambda k: abs(k - metric_value))
    return float(conversion_map[nearest_key])


def detect_and_convert_snellen_highlight(token: str) -> float:
    """
    Parse a Snellen optotype token (metric or imperial) and convert to imperial.
    Handles formats like '6', '20', '7_5' (7.5 metric), etc.
    """
    metric_values = {60, 30, 20, 15, 12, 9, 7.5, 6, 5, 4, 3}
    # Replace underscore with dot for decimal parsing (e.g., '7_5' -> '7.5')
    token_normalized = token.replace("_", ".")
    try:
        value = float(token_normalized)
    except ValueError:
        return 999.0
    
    # Heuristic: if the value is in the known metric set, it's metric; otherwise assume imperial.
    # Special case: single digits < 6 are ambiguous (could be part of a decimal or 20/5, 20/4, etc.)
    # Return 999 to signal caller to try combining with adjacent token
    if value in metric_values:
        return metric_to_imperial(value)
    elif 0 < value < 6:
        # Single digit < 6 could be part of decimal; return 999 to defer to fallback
        return 999.0
    else:
        # Already imperial or unknown; return as-is
        return value


def is_snellen(chart: str) -> bool:
    return chart.startswith("snellen_chart")


def is_echart(chart: str) -> bool:
    return chart.startswith("echart")


def is_number_chart(chart: str) -> bool:
    return chart.startswith("number_chart")


def get_question(row: dict, config: dict) -> str:
    chart = normalize_chart(row.get("Chart_Display"))
    occ = (row.get("Occluder_State") or "").strip()

    # JCC chart with occlusion is an error state
    if chart == "jcc_chart" and (occ == "Left_Occluded" or occ == "Right_Occluded"):
        return "INTERMITTENT ERROR"

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

    if is_echart(chart):
        return config["question_echart"]

    return config["question_fallback"]


def get_snellen_base_and_highlight(chart_name: str) -> tuple[str, float]:
    chart = normalize_chart(chart_name)
    if not is_snellen(chart):
        return ("", 999.0)
    parts = chart.split("_")
    if len(parts) < 4:
        return (chart, 999.0)
    # Handle case where the last highlight might be a decimal like 7_5 (representing 7.5)
    # Check if last token + next-to-last token can form a valid decimal (e.g., "7_5")
    last_token = parts[-1]
    second_last_token = parts[-2] if len(parts) > 4 else None

    # If the last value repeats one of the previous values, treat it as the highlighted optotype
    # e.g., snellen_chart_20_15_15 -> base snellen_chart_20_15, highlight 15
    #       snellen_chart_70_60_50_60 -> base snellen_chart_70_60_50, highlight 60
    if last_token in parts[2:-1]:
        return ("snellen_chart_" + "_".join(parts[2:-1]), detect_and_convert_snellen_highlight(last_token))
    
    # Priority: Try combining last two tokens if second-to-last is a digit
    # This catches cases like snellen_chart_60_30_20_7_5 where 7_5 should be 7.5
    if second_last_token and second_last_token.isdigit():
        try:
            combined_highlight = detect_and_convert_snellen_highlight(f"{second_last_token}_{last_token}")
            # Only accept combined if it's a known metric decimal (like 7_5 -> 7.5 -> 25 imperial)
            combined_as_float = float(f"{second_last_token}.{last_token}")
            metric_decimals = {7.5}  # Known metric decimals
            if combined_as_float in metric_decimals and combined_highlight < 999.0:
                return ("snellen_chart_" + "_".join(parts[2:-2]), combined_highlight)
        except (ValueError, IndexError):
            pass
    
    # Try parsing last token as optotype
    try:
        highlight = detect_and_convert_snellen_highlight(last_token)
        base = "snellen_chart_" + "_".join(parts[2:-1])
    except (ValueError, IndexError):
        highlight = 999.0
        base = chart
    
    return (base, highlight)


def has_sph_change(row_a: dict, row_b: dict) -> bool:
    try:
        a_r_sph = float(row_a.get("R_SPH") or 0)
        b_r_sph = float(row_b.get("R_SPH") or 0)
        a_l_sph = float(row_a.get("L_SPH") or 0)
        b_l_sph = float(row_b.get("L_SPH") or 0)
        return (abs(a_r_sph - b_r_sph) > 0.001) or (abs(a_l_sph - b_l_sph) > 0.001)
    except (ValueError, TypeError):
        return False


def get_snellen_answer(current: dict, previous: dict, nxt: dict, prevprev: dict) -> str:
    """Determine Snellen answer based on chart transition and SPH refinement."""
    current_chart = normalize_chart(current.get("Chart_Display"))
    current_base, current_highlight = get_snellen_base_and_highlight(current_chart)
    
    # If no previous row or previous is not Snellen, base answer on current chart alone
    if not previous:
        if abs(current_highlight - 20) < 0.001:
            return "Able to read."
        return "Blurry."
    
    previous_chart = normalize_chart(previous.get("Chart_Display"))
    
    # Only compare transitions if both current and previous are Snellen charts
    if not is_snellen(previous_chart):
        if abs(current_highlight - 20) < 0.001:
            return "Able to read."
        return "Blurry."

    previous_base, previous_highlight = get_snellen_base_and_highlight(previous_chart)

    sph_changed = has_sph_change(current, previous)

    if prevprev and abs(current_highlight - previous_highlight) < 0.001:
        prevprev_chart = normalize_chart(prevprev.get("Chart_Display"))
        if is_snellen(prevprev_chart):
            prevprev_base, prevprev_highlight = get_snellen_base_and_highlight(prevprev_chart)
            if prevprev_base == current_base and abs(prevprev_highlight - current_highlight) < 0.001:
                try:
                    cur_r_sph = float(current.get("R_SPH") or 0)
                    prev_r_sph = float(previous.get("R_SPH") or 0)
                    prevprev_r_sph = float(prevprev.get("R_SPH") or 0)
                    cur_l_sph = float(current.get("L_SPH") or 0)
                    prev_l_sph = float(previous.get("L_SPH") or 0)
                    prevprev_l_sph = float(prevprev.get("L_SPH") or 0)
                except (ValueError, TypeError):
                    cur_r_sph = prev_r_sph = prevprev_r_sph = 0.0
                    cur_l_sph = prev_l_sph = prevprev_l_sph = 0.0
                if (abs(cur_r_sph - prevprev_r_sph) < 0.001 and abs(cur_r_sph - prev_r_sph) > 0.001) or \
                   (abs(cur_l_sph - prevprev_l_sph) < 0.001 and abs(cur_l_sph - prev_l_sph) > 0.001):
                    return "Unable to read."

    if abs(current_highlight - previous_highlight) < 0.001 and sph_changed:
        if nxt and is_snellen(normalize_chart(nxt.get("Chart_Display"))):
            nxt_base, nxt_highlight = get_snellen_base_and_highlight(nxt.get("Chart_Display"))
            if nxt_base == current_base and abs(nxt_highlight - current_highlight) < 0.001 and has_sph_change(current, nxt):
                return "Unable to read."
        return "Getting better."

    if nxt and is_snellen(normalize_chart(nxt.get("Chart_Display"))):
        nxt_base, nxt_highlight = get_snellen_base_and_highlight(nxt.get("Chart_Display"))
        if nxt_base == current_base:
            if nxt_highlight < current_highlight:
                return "Able to read."
            if nxt_highlight > current_highlight:
                return "Unable to read."
            if abs(nxt_highlight - current_highlight) < 0.001 and has_sph_change(current, nxt):
                return "Unable to read."
        # If the next Snellen row moves to a finer highlight across a different base,
        # treat the current line as read easily.
        if nxt_highlight < current_highlight:
            return "Able to read."

    if current_highlight < previous_highlight:
        return "Able to read."

    if abs(current_highlight - previous_highlight) < 0.001 and not sph_changed:
        return "Blurry."

    if current_highlight > previous_highlight:
        return "Unable to read."

    if abs(current_highlight - 20) < 0.001:
        return "Able to read."
    
    return "Blurry."


def get_flip_answer(current: dict, nxt: dict) -> str:
    occ = current.get("Occluder_State") or ""
    if "Flip1" in occ:
        return "-"
    if not nxt:
        return ""

    if "Axis" in occ:
        # Determine which eye based on occluder state
        # If "Left" in occ (like "Left_Axis_Flip2"), use L_AXIS; otherwise use R_AXIS
        key = "L_AXIS" if "Left" in occ else "R_AXIS"
        eye_prefix = "LAM" if "Left" in occ else "RAM"  # LAM=Left Axis Movement, RAM=Right Axis Movement
        try:
            cur = float(current.get(key) or 0)
            nxt_val = float(nxt.get(key) or 0)
        except ValueError:
            return ""
        if nxt_val > cur:
            return f"Flip 1 - {eye_prefix} Axis"
        if nxt_val < cur:
            return f"Flip 2 - {eye_prefix} Axis"
        return "No change"

    if "Power" in occ:
        # Determine which eye based on occluder state
        # If "Left" in occ (like "Left_Power_Flip2"), use L_CYL; otherwise use R_CYL
        key = "L_CYL" if "Left" in occ else "R_CYL"
        eye_prefix = "LPM" if "Left" in occ else "RPM"  # LPM=Left Power Movement, RPM=Right Power Movement
        try:
            cur = float(current.get(key) or 0)
            nxt_val = float(nxt.get(key) or 0)
        except ValueError:
            return ""
        # Flip 1 = increase (more positive/less negative)
        # Flip 2 = decrease (more negative)
        if nxt_val > cur:
            return f"Flip 1 - {eye_prefix} Power"
        if nxt_val < cur:
            return f"Flip 2 - {eye_prefix} Power"
        return "No change"

    return ""


def get_answer(row: dict, prev: dict, prevprev: dict, nxt: dict, config: dict) -> str:
    chart = normalize_chart(row.get("Chart_Display"))
    occ = row.get("Occluder_State") or ""

    # JCC chart with occlusion is an error state
    if chart == "jcc_chart" and (occ == "Left_Occluded" or occ == "Right_Occluded"):
        return "INTERMITTENT ERROR"

    if "Flip1" in occ or "Flip2" in occ:
        return get_flip_answer(row, nxt)

    if is_snellen(chart):
        return get_snellen_answer(row, prev, nxt, prevprev)

    if chart == "duochrome":
        return config["answer_duochrome"]

    if chart == "jcc_chart":
        return config["answer_jcc"]

    if chart == "near_vision" or is_number_chart(chart):
        return config["answer_near"]

    if is_echart(chart) or is_snellen(chart) or chart == "pictorial_chart":
        return config["answer_distance"]

    return config["answer_fallback"]


def has_state_change_in_next(rows: list[dict], idx: int, current_row: dict, window: int = 4) -> bool:
    """Check if next rows have a state change (different occluder state, SPH, CYL, or AXIS)."""
    if idx + 1 >= len(rows):
        return False
    
    curr_occ = current_row.get("Occluder_State") or ""
    for j in range(idx + 1, min(len(rows), idx + 1 + window)):
        nxt_row = rows[j]
        nxt_occ = nxt_row.get("Occluder_State") or ""
        
        # Check if occluder state changed
        if nxt_occ != curr_occ:
            return True
        
        # Check if phoropter state changed (SPH, CYL, AXIS)
        try:
            if (abs(float(current_row.get("R_SPH") or 0) - float(nxt_row.get("R_SPH") or 0)) > 0.001 or
                abs(float(current_row.get("L_SPH") or 0) - float(nxt_row.get("L_SPH") or 0)) > 0.001 or
                abs(float(current_row.get("R_CYL") or 0) - float(nxt_row.get("R_CYL") or 0)) > 0.001 or
                abs(float(current_row.get("L_CYL") or 0) - float(nxt_row.get("L_CYL") or 0)) > 0.001 or
                abs(float(current_row.get("R_AXIS") or 0) - float(nxt_row.get("R_AXIS") or 0)) > 0.001 or
                abs(float(current_row.get("L_AXIS") or 0) - float(nxt_row.get("L_AXIS") or 0)) > 0.001):
                return True
        except (ValueError, TypeError):
            pass
    
    return False


def get_confidence(rows: list[dict], idx: int, question: str, occ: str, config: dict) -> str:
    # For Flip1: always confident (never confused)
    if "Flip1" in occ:
        return config["confidence_confident_label"]
    
    # For Flip2: if next row has a state change, mark as confident
    if "Flip2" in occ:
        if has_state_change_in_next(rows, idx, rows[idx]):
            return config["confidence_confident_label"]

    # Snellen highlight change or SPH adjustment on same chart base
    current_chart = normalize_chart(rows[idx].get("Chart_Display"))
    if is_snellen(current_chart):
        if idx >= 2:
            prevprev = rows[idx - 2]
            prev = rows[idx - 1]
            if is_snellen(normalize_chart(prev.get("Chart_Display"))) and is_snellen(normalize_chart(prevprev.get("Chart_Display"))):
                current_base, current_highlight = get_snellen_base_and_highlight(current_chart)
                prev_base, prev_highlight = get_snellen_base_and_highlight(prev.get("Chart_Display"))
                prevprev_base, prevprev_highlight = get_snellen_base_and_highlight(prevprev.get("Chart_Display"))
                if current_base == prev_base == prevprev_base and abs(current_highlight - prev_highlight) < 0.001 and abs(current_highlight - prevprev_highlight) < 0.001:
                    try:
                        cur_r_sph = float(rows[idx].get("R_SPH") or 0)
                        prev_r_sph = float(prev.get("R_SPH") or 0)
                        prevprev_r_sph = float(prevprev.get("R_SPH") or 0)
                        cur_l_sph = float(rows[idx].get("L_SPH") or 0)
                        prev_l_sph = float(prev.get("L_SPH") or 0)
                        prevprev_l_sph = float(prevprev.get("L_SPH") or 0)
                    except (ValueError, TypeError):
                        cur_r_sph = prev_r_sph = prevprev_r_sph = 0.0
                        cur_l_sph = prev_l_sph = prevprev_l_sph = 0.0
                    if (abs(cur_r_sph - prevprev_r_sph) < 0.001 and abs(cur_r_sph - prev_r_sph) > 0.001) or \
                       (abs(cur_l_sph - prevprev_l_sph) < 0.001 and abs(cur_l_sph - prev_l_sph) > 0.001):
                        return config["confidence_confused_label"]

        if idx + 1 < len(rows):
            nxt_chart = normalize_chart(rows[idx + 1].get("Chart_Display"))
            if is_snellen(nxt_chart):
                current_base, current_highlight = get_snellen_base_and_highlight(current_chart)
                next_base, next_highlight = get_snellen_base_and_highlight(nxt_chart)
                if current_base == next_base:
                    if next_highlight < current_highlight:
                        return config["confidence_confident_label"]
                    if abs(next_highlight - current_highlight) < 0.001 and has_sph_change(rows[idx], rows[idx + 1]):
                        return config["confidence_confident_label"]
                # If next Snellen row moves to a finer highlight across bases, mark confident
                if next_highlight < current_highlight:
                    return config["confidence_confident_label"]
    
    # Default logic: check if same question repeats in window
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
            prev = rows[idx - 1] if idx > 0 else None
            prevprev = rows[idx - 2] if idx > 1 else None
            nxt = rows[idx + 1] if idx + 1 < len(rows) else None
            question = get_question(row, config)
            answer = get_answer(row, prev, prevprev, nxt, config)
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
