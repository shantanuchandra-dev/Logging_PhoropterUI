"""
Count CSV files created today in a Supabase Storage bucket and send a
tabular phase-status report to a Google Chat space via incoming webhook.

Sessions as rows, phases as columns.

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

LEGEND_LINES = [
    "DV  = Distance Vision",
    "RER = Right Eye Refraction",
    "JAR = Jcc Axis Right",
    "JPR = Jcc Power Right",
    "DR  = Duochrome Right",
    "VR  = Validation Right",
    "LER = Left Eye Refraction",
    "JAL = Jcc Axis Left",
    "JPL = Jcc Power Left",
    "DL  = Duochrome Left",
    "VL  = Validation Left",
    "VDi = Validation Distance",
    "BB  = Binocular Balance",
    "NAR = Near Add Right",
    "NAL = Near Add Left",
    "NAB = Near Add Bino",
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


def get_phase_icon(phase_id: str, metadata: dict | None) -> str:
    if metadata is None:
        return "➖"
    if phase_id in set(metadata.get("phases_completed", [])):
        return "✅"
    if phase_id in set(metadata.get("phases_skipped", [])):
        return "❌"
    return "➖"


def build_table(session_ids: list[str], metadata_map: dict[str, dict]) -> str:
    """Build table: sessions as rows, phases as columns."""
    sid_col_width = 16
    phase_col_width = 4

    header = f"{'Session ID':<{sid_col_width}}"
    for _, short in ALL_PHASES:
        header += f" {short:^{phase_col_width}}"

    separator = "─" * len(header)

    rows = [header, separator]
    for sid in session_ids:
        meta = metadata_map.get(sid)
        row = f"{sid:<{sid_col_width}}"
        for phase_id, _ in ALL_PHASES:
            icon = get_phase_icon(phase_id, meta)
            row += f"  {icon}  "
        rows.append(row)

    return "\n".join(rows)


def build_card_payload(
    today_str: str,
    count: int,
    bucket: str,
    session_ids: list[str],
    metadata_map: dict[str, dict],
) -> dict:
    """Build a Google Chat Cards v2 payload with the phase-status table."""
    table_text = build_table(session_ids, metadata_map)
    legend_text = "\n".join(LEGEND_LINES)

    return {
        "cardsV2": [
            {
                "cardId": "dailyCsvReport",
                "card": {
                    "header": {
                        "title": f"Daily CSV Report — {today_str}",
                        "subtitle": f"{count} CSV file(s) created today  |  Bucket: {bucket}",
                    },
                    "sections": [
                        {
                            "header": "Phase Status  (✅ = Completed, ❌ = Skipped)",
                            "widgets": [
                                {
                                    "textParagraph": {
                                        "text": f"<pre>{table_text}</pre>",
                                    }
                                }
                            ],
                        },
                        {
                            "header": "Legend",
                            "collapsible": True,
                            "widgets": [
                                {
                                    "textParagraph": {
                                        "text": f"<pre>{legend_text}</pre>",
                                    }
                                }
                            ],
                        },
                    ],
                },
            }
        ]
    }


def build_empty_card(today_str: str, bucket: str) -> dict:
    return {
        "cardsV2": [
            {
                "cardId": "dailyCsvReport",
                "card": {
                    "header": {
                        "title": f"Daily CSV Report — {today_str}",
                        "subtitle": f"0 CSV files created today  |  Bucket: {bucket}",
                    },
                    "sections": [
                        {
                            "widgets": [
                                {
                                    "textParagraph": {
                                        "text": "No CSV files were created today.",
                                    }
                                }
                            ]
                        }
                    ],
                },
            }
        ]
    }


def send_google_chat_card(webhook_url: str, payload: dict) -> None:
    resp = requests.post(webhook_url, json=payload, timeout=30)
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

    if session_ids:
        metadata_map: dict[str, dict] = {}
        for sid in session_ids:
            meta = fetch_metadata(storage, sid)
            if meta:
                metadata_map[sid] = meta

        payload = build_card_payload(today_str, count, bucket, session_ids, metadata_map)
    else:
        payload = build_empty_card(today_str, bucket)

    send_google_chat_card(webhook_url, payload)


if __name__ == "__main__":
    main()
