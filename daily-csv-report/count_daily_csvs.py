"""
Count CSV files created today in a Supabase Storage bucket and send the count
to a Google Chat space via incoming webhook.

Required env vars:
    SUPABASE_URL, SUPABASE_SERVICE_KEY, GOOGLE_CHAT_WEBHOOK_URL
Optional:
    SUPABASE_BUCKET (default: eye-test-sessions)
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import requests
from supabase import create_client


def get_env(name: str, default: str | None = None, required: bool = True) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        if default is not None:
            return default
        if required:
            print(f"ERROR: Missing required environment variable: {name}")
            sys.exit(1)
    return value


def list_todays_csvs(supabase_url: str, supabase_key: str, bucket: str) -> list[str]:
    """Return names of .csv files created today (UTC) in the bucket."""
    client = create_client(supabase_url, supabase_key)
    storage = client.storage.from_(bucket)

    today = datetime.now(timezone.utc).date()
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


def send_google_chat_message(webhook_url: str, text: str) -> None:
    resp = requests.post(webhook_url, json={"text": text}, timeout=30)
    resp.raise_for_status()
    print("Google Chat notification sent successfully.")


def main() -> None:
    supabase_url = get_env("SUPABASE_URL")
    supabase_key = get_env("SUPABASE_SERVICE_KEY")
    bucket = get_env("SUPABASE_BUCKET", default="eye-test-sessions")
    webhook_url = get_env("GOOGLE_CHAT_WEBHOOK_URL")

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"Checking bucket '{bucket}' for CSV files created on {today_str} ...")
    csv_files = list_todays_csvs(supabase_url, supabase_key, bucket)

    count = len(csv_files)
    print(f"Found {count} CSV file(s) created today.")

    message_lines = [
        f"*Daily CSV Report — {today_str}*",
        "─" * 30,
        f"New CSV files created today: *{count}*",
        f"Bucket: `{bucket}`",
    ]
    if csv_files:
        message_lines.append("")
        message_lines.append("Files:")
        for name in sorted(csv_files):
            message_lines.append(f"  • {name}")

    message = "\n".join(message_lines)
    send_google_chat_message(webhook_url, message)


if __name__ == "__main__":
    main()
