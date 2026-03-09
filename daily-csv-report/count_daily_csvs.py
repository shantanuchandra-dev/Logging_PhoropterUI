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


def get_phase_icon(phase_id: str, metadata: dict | None) -> str:
    if metadata is None:
        return "➖"
    if phase_id in set(metadata.get("phases_completed", [])):
        return "✅"
    if phase_id in set(metadata.get("phases_skipped", [])):
        return "❌"
    return "➖"


def build_phase_rows(session_ids: list[str], metadata_map: dict[str, dict]) -> list[dict]:
    """Build phase status data for the card widgets."""
    rows = []
    for phase_id, phase_label in ALL_PHASES:
        icons = []
        for sid in session_ids:
            meta = metadata_map.get(sid)
            icons.append(get_phase_icon(phase_id, meta))
        rows.append({"label": phase_label, "icons": icons})
    return rows


def build_card_payload(
    today_str: str,
    count: int,
    bucket: str,
    session_ids: list[str],
    metadata_map: dict[str, dict],
) -> dict:
    """Build a Google Chat Cards v2 payload with the phase-status table."""
    phase_rows = build_phase_rows(session_ids, metadata_map)

    short_ids = [sid[-6:] for sid in session_ids]
    header_row = f"{'Test Step':<{PHASE_LABEL_WIDTH}}  " + "  ".join(
        f"{sid:>6}" for sid in short_ids
    )
    separator = "─" * len(header_row)

    table_lines = [header_row, separator]
    for row in phase_rows:
        label = f"{row['label']:<{PHASE_LABEL_WIDTH}}"
        icons_str = "  ".join(f"  {icon}   " for icon in row["icons"])
        table_lines.append(f"{label}  {icons_str}")

    session_list = "\n".join(
        f"• <b>...{sid[-6:]}</b> → {sid}" for sid in session_ids
    )

    card = {
        "cardsV2": [
            {
                "cardId": "dailyCsvReport",
                "card": {
                    "header": {
                        "title": f"📊 Daily CSV Report — {today_str}",
                        "subtitle": f"{count} CSV file(s) created today  |  Bucket: {bucket}",
                    },
                    "sections": [
                        {
                            "header": "Session IDs",
                            "collapsible": True,
                            "widgets": [
                                {
                                    "textParagraph": {
                                        "text": session_list,
                                    }
                                }
                            ],
                        },
                        {
                            "header": "Phase Status",
                            "widgets": [
                                {
                                    "textParagraph": {
                                        "text": "<pre>" + "\n".join(table_lines) + "</pre>",
                                    }
                                }
                            ],
                        },
                    ],
                },
            }
        ]
    }
    return card


def build_empty_card(today_str: str, bucket: str) -> dict:
    return {
        "cardsV2": [
            {
                "cardId": "dailyCsvReport",
                "card": {
                    "header": {
                        "title": f"📊 Daily CSV Report — {today_str}",
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
