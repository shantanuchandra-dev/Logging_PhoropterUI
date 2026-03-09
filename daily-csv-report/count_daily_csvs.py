"""
Count CSV files created today in a Supabase Storage bucket and send a
tabular phase-status report to Google Chat via webhook.

Sessions as rows, phases as columns, plain text format.

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
    ("distance_vision", "DV"),
    ("right_eye_refraction", "RER"),
    ("jcc_axis_right", "JAR"),
    ("jcc_power_right", "JPR"),
    ("duochrome_right", "DR"),
    ("validation_right", "VR"),
    ("left_eye_refraction", "LER"),
    ("jcc_axis_left", "JAL"),
    ("jcc_power_left", "JPL"),
    ("duochrome_left", "DL"),
    ("validation_left", "VL"),
    ("validation_distance", "VDi"),
    ("binocular_balance", "BB"),
    ("near_add_right", "NAR"),
    ("near_add_left", "NAL"),
    ("near_add_bino", "NAB"),
]


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
    meta_path = f"{session_id}_metadata.json"
    try:
        data = storage.download(meta_path)
        return json.loads(data)
    except Exception:
        return None


def get_phase_icon(phase_id: str, metadata: dict | None) -> str:
    if metadata is None:
        return "  -"
    if phase_id in set(metadata.get("phases_completed", [])):
        return "  ✅"
    if phase_id in set(metadata.get("phases_skipped", [])):
        return "  ❌"
    return "  -"


def build_table(session_ids: list[str], metadata_map: dict[str, dict]) -> str:
    """Build a clean space-aligned table, no pipes, no markdown."""
    sid_width = 16
    col_width = 5

    header = f"{'Session ID':<{sid_width}}"
    for _, short in ALL_PHASES:
        header += f"{short:>{col_width}}"

    rows = [header]
    for sid in session_ids:
        meta = metadata_map.get(sid)
        row = f"{sid:<{sid_width}}"
        for phase_id, _ in ALL_PHASES:
            row += get_phase_icon(phase_id, meta)
        rows.append(row)

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

    header = f"*Daily CSV Report — {today_str}*\n"
    header += f"CSV files created today: *{count}*\n"
    header += f"Bucket: {bucket}\n"
    header += "✅ = Completed    ❌ = Skipped\n\n"

    if session_ids:
        table = build_table(session_ids, metadata_map)
        legend = "\nDV=Distance Vision  RER=Right Eye Refraction  JAR=Jcc Axis Right  JPR=Jcc Power Right  DR=Duochrome Right  VR=Validation Right  LER=Left Eye Refraction  JAL=Jcc Axis Left  JPL=Jcc Power Left  DL=Duochrome Left  VL=Validation Left  VDi=Validation Distance  BB=Binocular Balance  NAR=Near Add Right  NAL=Near Add Left  NAB=Near Add Bino"
        message = header + "```\n" + table + "\n```" + legend
    else:
        message = header + "No CSV files were created today."

    send_google_chat_message(webhook_url, message)


if __name__ == "__main__":
    main()
