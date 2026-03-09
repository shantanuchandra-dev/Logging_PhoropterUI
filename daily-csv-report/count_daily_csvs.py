"""
Count CSV files created today in a Supabase Storage bucket and send a
readable phase-status report to Google Chat via webhook.

Each session displayed as a card section with phase statuses.

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


def format_session_phases(metadata: dict | None) -> str:
    """Format all phases for one session as readable HTML lines."""
    if metadata is None:
        return "(metadata not found)"

    completed = set(metadata.get("phases_completed", []))
    skipped = set(metadata.get("phases_skipped", []))

    completed_list = []
    skipped_list = []

    for phase_id, phase_name in ALL_PHASES:
        if phase_id in completed:
            completed_list.append(phase_name)
        elif phase_id in skipped:
            skipped_list.append(phase_name)

    lines = []
    if completed_list:
        lines.append(f"<b>✅ Completed ({len(completed_list)}):</b>")
        lines.append(", ".join(completed_list))
    if skipped_list:
        lines.append(f"<b>❌ Skipped ({len(skipped_list)}):</b>")
        lines.append(", ".join(skipped_list))

    return "<br>".join(lines)


def build_card(
    today_str: str,
    count: int,
    bucket: str,
    session_ids: list[str],
    metadata_map: dict[str, dict],
) -> dict:
    sections = []

    for sid in session_ids:
        meta = metadata_map.get(sid)
        completed = set((meta or {}).get("phases_completed", []))
        total = len(ALL_PHASES)
        done = len([p for p, _ in ALL_PHASES if p in completed])

        sections.append({
            "header": f"📄 Session {sid}    ({done}/{total} phases completed)",
            "collapsible": True,
            "uncollapsibleWidgetsCount": 1,
            "widgets": [
                {
                    "textParagraph": {
                        "text": format_session_phases(meta),
                    }
                }
            ],
        })

    return {
        "cardsV2": [
            {
                "cardId": "dailyCsvReport",
                "card": {
                    "header": {
                        "title": f"Daily CSV Report — {today_str}",
                        "subtitle": f"{count} CSV file(s) created today  •  Bucket: {bucket}",
                    },
                    "sections": sections,
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
                        "subtitle": f"0 CSV files created today  •  Bucket: {bucket}",
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

        payload = build_card(today_str, count, bucket, session_ids, metadata_map)
    else:
        payload = build_empty_card(today_str, bucket)

    send_google_chat_card(webhook_url, payload)


if __name__ == "__main__":
    main()
