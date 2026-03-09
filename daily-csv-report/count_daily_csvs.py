"""
Count CSV files created today in a Supabase Storage bucket and send a
visual phase-status table (as an image) to Google Chat via webhook.

Sessions as rows, phases as columns, rendered as a proper table image.

Required env vars:
    SUPABASE_URL, SUPABASE_SERVICE_KEY, GOOGLE_CHAT_WEBHOOK_URL
Optional:
    SUPABASE_BUCKET (default: eye-test-sessions)
"""
from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from supabase import create_client

ALL_PHASES = [
    ("distance_vision", "Distance\nVision"),
    ("right_eye_refraction", "Right Eye\nRefraction"),
    ("jcc_axis_right", "Jcc Axis\nRight"),
    ("jcc_power_right", "Jcc Power\nRight"),
    ("duochrome_right", "Duochrome\nRight"),
    ("validation_right", "Validation\nRight"),
    ("left_eye_refraction", "Left Eye\nRefraction"),
    ("jcc_axis_left", "Jcc Axis\nLeft"),
    ("jcc_power_left", "Jcc Power\nLeft"),
    ("duochrome_left", "Duochrome\nLeft"),
    ("validation_left", "Validation\nLeft"),
    ("validation_distance", "Validation\nDistance"),
    ("binocular_balance", "Binocular\nBalance"),
    ("near_add_right", "Near Add\nRight"),
    ("near_add_left", "Near Add\nLeft"),
    ("near_add_bino", "Near Add\nBino"),
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


def get_phase_status(phase_id: str, metadata: dict | None) -> str:
    if metadata is None:
        return "skip"
    if phase_id in set(metadata.get("phases_completed", [])):
        return "done"
    if phase_id in set(metadata.get("phases_skipped", [])):
        return "skip"
    return "none"


def render_table_image(
    session_ids: list[str],
    metadata_map: dict[str, dict],
    today_str: str,
    count: int,
) -> bytes:
    """Render the phase-status table as a PNG image and return bytes."""
    col_headers = ["Session ID"] + [label for _, label in ALL_PHASES]
    n_rows = len(session_ids)
    n_cols = len(col_headers)

    cell_data = []
    cell_colors = []

    header_bg = "#2d2d2d"
    header_text = "white"
    row_bg_even = "#1a1a1a"
    row_bg_odd = "#252525"
    done_color = "#27ae60"
    skip_color = "#e74c3c"
    none_color = "#555555"

    for i, sid in enumerate(session_ids):
        meta = metadata_map.get(sid)
        row_bg = row_bg_even if i % 2 == 0 else row_bg_odd
        row = [sid]
        colors = [row_bg]
        for phase_id, _ in ALL_PHASES:
            status = get_phase_status(phase_id, meta)
            row.append("✓" if status == "done" else "✗" if status == "skip" else "−")
            colors.append(row_bg)
        cell_data.append(row)
        cell_colors.append(colors)

    fig_width = max(18, n_cols * 1.1)
    fig_height = max(3, (n_rows + 1) * 0.55 + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor("#121212")
    ax.set_facecolor("#121212")
    ax.axis("off")

    title = f"Daily CSV Report — {today_str}    ({count} sessions)"
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=20, loc="left")

    table = ax.table(
        cellText=cell_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_linewidth(0.5)

        if row == 0:
            cell.set_facecolor(header_bg)
            cell.get_text().set_color(header_text)
            cell.get_text().set_fontsize(8)
            cell.get_text().set_fontweight("bold")
        else:
            bg = cell_colors[row - 1][col]
            cell.set_facecolor(bg)

            text_val = cell.get_text().get_text()
            if col == 0:
                cell.get_text().set_color("white")
                cell.get_text().set_fontsize(8)
                cell.get_text().set_ha("left")
            elif text_val == "✓":
                cell.get_text().set_color(done_color)
                cell.get_text().set_fontsize(13)
                cell.get_text().set_fontweight("bold")
            elif text_val == "✗":
                cell.get_text().set_color(skip_color)
                cell.get_text().set_fontsize(13)
                cell.get_text().set_fontweight("bold")
            else:
                cell.get_text().set_color(none_color)
                cell.get_text().set_fontsize(11)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def upload_image(storage, image_bytes: bytes, today_str: str) -> str:
    """Upload PNG to Supabase Storage and return a signed URL."""
    path = f"reports/daily_report_{today_str}.png"
    try:
        storage.remove([path])
    except Exception:
        pass
    storage.upload(
        path,
        image_bytes,
        {"content-type": "image/png", "upsert": "true"},
    )
    signed = storage.create_signed_url(path, 60 * 60 * 24 * 7)
    return signed["signedURL"]


def send_google_chat_card(webhook_url: str, image_url: str, today_str: str, count: int, bucket: str) -> None:
    payload = {
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
                            "widgets": [
                                {
                                    "image": {
                                        "imageUrl": image_url,
                                        "altText": f"Daily CSV phase status report for {today_str}",
                                    }
                                }
                            ]
                        }
                    ],
                },
            }
        ]
    }
    resp = requests.post(webhook_url, json=payload, timeout=30)
    resp.raise_for_status()
    print("Google Chat notification sent successfully.")


def send_empty_card(webhook_url: str, today_str: str, bucket: str) -> None:
    payload = {
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

        print("Rendering table image ...")
        image_bytes = render_table_image(session_ids, metadata_map, today_str, count)

        print("Uploading image to Supabase Storage ...")
        image_url = upload_image(storage, image_bytes, today_str)
        print(f"Image URL: {image_url}")

        send_google_chat_card(webhook_url, image_url, today_str, count, bucket)
    else:
        send_empty_card(webhook_url, today_str, bucket)


if __name__ == "__main__":
    main()
