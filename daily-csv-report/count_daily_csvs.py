"""
Count CSV files created today in a Supabase Storage bucket and send a
tabular phase-status report to a Google Chat space via incoming webhook.

Required env vars:
    SUPABASE_URL, SUPABASE_SERVICE_KEY, GOOGLE_CHAT_WEBHOOK_URL
Optional:
    SUPABASE_BUCKET (default: eye-test-sessions)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import requests
from supabase import create_client

ALL_PHASES = [
    ("distance_vision", "Distance Vision"),
    ("right_eye_refraction", "Right Eye Refraction"),
    ("jcc_axis_right", "Jcc Axis Right"),
    ("jcc_power_right", "Jcc Power Right"),
    ("duochrome_right", "Duochrome Right"),
    ("validation_right", "Validation Right"),
    ("left_eye_refraction", "Left Eye Refraction"),
    ("jcc_axis_left", "Jcc Axis Left"),
    ("jcc_power_left", "Jcc Power Left"),
    ("duochrome_left", "Duochrome Left"),
    ("validation_left", "Validation Left"),
    ("validation_distance", "Validation Distance"),
    ("binocular_balance", "Binocular Balance"),
    ("near_add_right", "Near Add Right"),
    ("near_add_left", "Near Add Left"),
    ("near_add_bino", "Near Add Bino"),
]

PHASE_LABEL_WIDTH = max(len(label) for _, label in ALL_PHASES)


def get_env(name: str, default: str | None = None, required: bool = True) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        if default is not None:
            return default
        if required:
            print(f"ERROR: Missing required environment variable: {name}")
            sys.exit(1)
    return value


def list_todays_csvs(storage, today) -> list[str]:
    """Return names of .csv files created today (UTC) in the bucket."""
    csv_files: list[str] = []
    items = storage.list()
    for item in items:
        name = item.get("name", "")
        if not name.lower().endswith(".csv"):
            continue
        created_at = item.get("created_at")
        if not created_at:
            continue
        file_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
        if file_date == today:
            csv_files.append(name)
    return csv_files


def fetch_metadata(storage, session_id: str) -> dict | None:
    """Download and parse the metadata JSON for a session."""
    meta_path = f"{session_id}_metadata.json"
    try:
        data = storage.download(meta_path)
        return json.loads(data)
    except Exception:
        return None


def build_table(session_ids: list[str], metadata_map: dict[str, dict]) -> str:
    """Build a monospace text table: phases as rows, sessions as columns."""
    col_width = max((len(sid) for sid in session_ids), default=6)
    col_width = max(col_width, 4)

    header_label = "Test Step".ljust(PHASE_LABEL_WIDTH)
    header_cols = " | ".join(sid.center(col_width) for sid in session_ids)
    header = f"| {header_label} | {header_cols} |"

    separator_label = "-" * PHASE_LABEL_WIDTH
    separator_cols = " | ".join("-" * col_width for _ in session_ids)
    separator = f"| {separator_label} | {separator_cols} |"

    rows: list[str] = [header, separator]

    for phase_id, phase_label in ALL_PHASES:
        label = phase_label.ljust(PHASE_LABEL_WIDTH)
        cells: list[str] = []
        for sid in session_ids:
            meta = metadata_map.get(sid)
            if meta is None:
                icon = "➖"
            elif phase_id in set(meta.get("phases_completed", [])):
                icon = "✅"
            elif phase_id in set(meta.get("phases_skipped", [])):
                icon = "❌"
            else:
                icon = "➖"
            cells.append(icon.center(col_width))
        row_cols = " | ".join(cells)
        rows.append(f"| {label} | {row_cols} |")

    return "\n".join(rows)


def send_google_chat_message(webhook_url: str, text: str) -> None:
    resp = requests.post(webhook_url, json={"text": text}, timeout=30)
    resp.raise_for_status()
    print("Google Chat notification sent successfully.")


def main() -> None:
    supabase_url = get_env("SUPABASE_URL")
    supabase_key = get_env("SUPABASE_SERVICE_KEY")
    bucket = get_env("SUPABASE_BUCKET", default="eye-test-sessions")
    webhook_url = get_env("GOOGLE_CHAT_WEBHOOK_URL")

    client = create_client(supabase_url, supabase_key)
    storage = client.storage.from_(bucket)

    today = datetime.now(timezone.utc).date()
    today_str = today.strftime("%Y-%m-%d")

    print(f"Checking bucket '{bucket}' for CSV files created on {today_str} ...")
    csv_files = list_todays_csvs(storage, today)

    count = len(csv_files)
    print(f"Found {count} CSV file(s) created today.")

    session_ids = sorted(name.removesuffix(".csv") for name in csv_files)

    metadata_map: dict[str, dict] = {}
    for sid in session_ids:
        meta = fetch_metadata(storage, sid)
        if meta:
            metadata_map[sid] = meta

    message_lines = [
        f"*Daily CSV Report — {today_str}*",
        f"New CSV files created today: *{count}*",
        f"Bucket: `{bucket}`",
        "",
    ]

    if session_ids:
        message_lines.append("```")
        message_lines.append(build_table(session_ids, metadata_map))
        message_lines.append("```")
    else:
        message_lines.append("No CSV files created today.")

    message = "\n".join(message_lines)
    send_google_chat_message(webhook_url, message)


if __name__ == "__main__":
    main()
