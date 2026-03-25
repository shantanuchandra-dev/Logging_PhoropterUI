"""
Count CSV files created in the last N days in a Supabase Storage bucket
and send a report showing per-day counts and skipped phases per session
to Google Chat via webhook.

Required env vars:
    SUPABASE_URL, SUPABASE_SERVICE_KEY, GOOGLE_CHAT_WEBHOOK_URL
Optional:
    SUPABASE_BUCKET (default: eye-test-sessions)
    LOOKBACK_DAYS  (default: 5)
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests
from supabase import create_client

ALL_PHASES = {
    "distance_vision": "Distance Vision",
    "right_eye_refraction": "Right Eye Refraction",
    "jcc_axis_right": "Jcc Axis Right",
    "jcc_power_right": "Jcc Power Right",
    "duochrome_right": "Duochrome Right",
    "validation_right": "Validation Right",
    "left_eye_refraction": "Left Eye Refraction",
    "jcc_axis_left": "Jcc Axis Left",
    "jcc_power_left": "Jcc Power Left",
    "duochrome_left": "Duochrome Left",
    "validation_left": "Validation Left",
    "validation_distance": "Validation Distance",
    "binocular_balance": "Binocular Balance",
    "near_add_right": "Near Add Right",
    "near_add_left": "Near Add Left",
    "near_add_bino": "Near Add Bino",
}


def get_env(name: str, default: str | None = None, required: bool = True) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        if default is not None:
            return default
        if required:
            print(f"ERROR: Missing required environment variable: {name}")
            sys.exit(1)
    return value


def list_csvs_in_range(storage, start_date, end_date) -> dict[str, list[str]]:
    """Return {date_str: [filename, ...]} for CSVs created between start_date and end_date (inclusive)."""
    by_date: dict[str, list[str]] = defaultdict(list)
    page_limit = 1000
    offset = 0

    while True:
        items = storage.list(
            path="",
            options={
                "limit": page_limit,
                "offset": offset,
                "sortBy": {"column": "created_at", "order": "desc"},
            },
        )
        if not items:
            break

        for item in items:
            name = item.get("name", "")
            if not name.lower().endswith(".csv"):
                continue
            created_at = item.get("created_at")
            if not created_at:
                continue
            file_date = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            ).date()
            if start_date <= file_date <= end_date:
                by_date[file_date.strftime("%Y-%m-%d")].append(name)
            elif file_date < start_date:
                return dict(by_date)

        if len(items) < page_limit:
            break
        offset += page_limit

    return dict(by_date)


def fetch_metadata(storage, session_id: str) -> dict | None:
    meta_path = f"{session_id}_metadata.json"
    try:
        data = storage.download(meta_path)
        return json.loads(data)
    except Exception:
        return None


def build_message(
    today_str: str,
    lookback_days: int,
    bucket: str,
    csvs_by_date: dict[str, list[str]],
    metadata_map: dict[str, dict],
) -> str:
    total = sum(len(v) for v in csvs_by_date.values())
    lines = [
        f"*Daily CSV Report — {today_str}*",
        f"Bucket: {bucket}",
        f"Period: last {lookback_days} days | Total CSVs: *{total}*",
        "",
        "─── Per-day breakdown ───",
    ]

    today = datetime.strptime(today_str, "%Y-%m-%d").date()
    for i in range(lookback_days):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        day_files = csvs_by_date.get(d, [])
        count = len(day_files)
        label = " (today)" if i == 0 else ""
        lines.append(f"  {d}{label}: *{count}* CSV(s)")

    today_files = csvs_by_date.get(today_str, [])
    today_session_ids = sorted(
        name.removesuffix(".csv") for name in today_files
    )

    if today_session_ids:
        lines.append("")
        lines.append(f"─── Today's sessions ({len(today_session_ids)}) ───")
        for sid in today_session_ids:
            meta = metadata_map.get(sid)
            lines.append(f"📄 {sid}.csv")
            if meta is None:
                lines.append("    (metadata not found)")
                continue
            skipped = meta.get("phases_skipped", [])
            if skipped:
                for phase_id in skipped:
                    name = ALL_PHASES.get(phase_id, phase_id)
                    lines.append(f"    ❌ {name}")
            else:
                lines.append("    ✅ All phases completed")
    else:
        lines.append("")
        lines.append("No CSV files created today.")

    return "\n".join(lines)


def send_google_chat_message(webhook_url: str, text: str) -> None:
    resp = requests.post(webhook_url, json={"text": text}, timeout=30)
    resp.raise_for_status()
    print("Google Chat notification sent successfully.")


def main() -> None:
    supabase_url = get_env("SUPABASE_URL")
    supabase_key = get_env("SUPABASE_SERVICE_KEY")
    bucket = get_env("SUPABASE_BUCKET", default="eye-test-sessions")
    webhook_url = get_env("GOOGLE_CHAT_WEBHOOK_URL")
    lookback_days = int(get_env("LOOKBACK_DAYS", default="5", required=False))

    client = create_client(supabase_url, supabase_key)
    storage = client.storage.from_(bucket)

    today = datetime.now(timezone.utc).date()
    today_str = today.strftime("%Y-%m-%d")
    start_date = today - timedelta(days=lookback_days - 1)

    print(f"Checking bucket '{bucket}' for CSVs from {start_date} to {today_str} ...")
    csvs_by_date = list_csvs_in_range(storage, start_date, today)

    total = sum(len(v) for v in csvs_by_date.values())
    print(f"Found {total} CSV file(s) across {lookback_days} day(s).")

    today_files = csvs_by_date.get(today_str, [])
    today_session_ids = sorted(
        name.removesuffix(".csv") for name in today_files
    )

    metadata_map: dict[str, dict] = {}
    for sid in today_session_ids:
        meta = fetch_metadata(storage, sid)
        if meta:
            metadata_map[sid] = meta

    message = build_message(
        today_str, lookback_days, bucket, csvs_by_date, metadata_map
    )
    send_google_chat_message(webhook_url, message)


if __name__ == "__main__":
    main()
