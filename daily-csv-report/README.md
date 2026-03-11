# Daily CSV Report

Counts CSV files created today in a Supabase Storage bucket and posts a
summary to Google Chat. Runs automatically via GitHub Actions.

**Files for this feature:**
- `daily-csv-report/count_daily_csvs.py` — the Python script
- `daily-csv-report/README.md` — this file
- `.github/workflows/daily-csv-count.yml` — the workflow (must live here for GitHub to detect it)

---

## Prerequisites

- A GitHub repository with Actions enabled
- A Supabase project with a Storage bucket (default: `eye-test-sessions`)
- A Google Chat space where you want to receive notifications

---

## 1. Create a Google Chat Webhook

1. Open **Google Chat** and navigate to the target **Space**.
2. Click the space name → **Apps & integrations** → **Webhooks**.
3. Click **Create a webhook**, give it a name (e.g., "CSV Daily Report").
4. Copy the generated webhook URL — you will need it in the next step.

---

## 2. Add GitHub Repository Secrets

Go to your repository on GitHub → **Settings** → **Secrets and variables** →
**Actions** → **New repository secret**, and add:

| Secret name              | Value                                                    |
| ------------------------ | -------------------------------------------------------- |
| `SUPABASE_URL`           | Your Supabase project URL (e.g. `https://xxx.supabase.co`) |
| `SUPABASE_SERVICE_KEY`   | Supabase **service_role** key (not the anon key)         |
| `SUPABASE_BUCKET`        | *(optional)* Bucket name; defaults to `eye-test-sessions` |
| `GOOGLE_CHAT_WEBHOOK_URL`| The webhook URL copied from Google Chat                  |

---

## 3. How It Works

- **Schedule**: The workflow runs daily at **18:00 UTC** (~11:30 PM IST).
  Edit the cron in `.github/workflows/daily-csv-count.yml` to change the time.
- **Manual trigger**: You can also run it on-demand from the GitHub Actions tab
  using the "Run workflow" button.
- The workflow:
  1. Checks out the repo.
  2. Installs Python and dependencies (`supabase`, `requests`).
  3. Runs `daily-csv-report/count_daily_csvs.py`, which lists the bucket,
     filters for `.csv` files created today (UTC), and POSTs the count to
     Google Chat.

---

## 4. Customisation

### Change the schedule

Edit the `cron` value in `.github/workflows/daily-csv-count.yml`:

```yaml
schedule:
  - cron: "0 18 * * *"   # <-- change this (UTC time)
```

Use [crontab.guru](https://crontab.guru/) to build your expression.

### Change the message format

Edit the `message_lines` list in `daily-csv-report/count_daily_csvs.py`.

### Change the bucket

Set the `SUPABASE_BUCKET` secret, or update the default in the script.
