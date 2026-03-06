# Save session CSVs to Supabase Storage

When a session is ended with **store** enabled, the backend writes session CSV and metadata locally and can upload the same files to **Supabase Storage**. You provide your Supabase project URL, service role key, and a storage bucket name.

---

## 1. Create a Supabase project

1. Go to [supabase.com](https://supabase.com) and sign in.
2. Create a new project (or use an existing one).
3. Wait for the project to finish setting up.

---

## 2. Create a Storage bucket

1. In the Supabase dashboard, open **Storage** in the left sidebar.
2. Click **New bucket**.
3. Name it e.g. `eye-test-sessions` (or any name; set `SUPABASE_BUCKET` in `.env` to match).
4. Choose **Public** or **Private** (for private, the service role key can still upload).
5. Click **Create bucket**.

---

## 3. Get your API credentials

1. Go to **Project Settings** (gear icon) → **API**.
2. Copy:
   - **Project URL** → use as `SUPABASE_URL` (e.g. `https://xxxx.supabase.co`).
   - **service_role** key (under "Project API keys") → use as `SUPABASE_SERVICE_KEY` or `SUPABASE_KEY`.  
   Use the **service_role** key (secret), not the anon key, so the server can upload to Storage.

---

## 4. Environment variables (use a `.env` file)

Define these in a **`.env`** file in the `eye_test_engine` folder. A template is in **`.env.example`**.

1. Copy the example:
   ```bash
   cd eye_test_engine
   cp .env.example .env
   ```
2. Edit `.env` and set your values. Do not commit `.env`; it is in `.gitignore`.

| Variable | Required | Description |
|----------|----------|-------------|
| `REMOTE_STORAGE` | Yes | Set to `supabase` to enable uploads. |
| `SUPABASE_URL` | Yes | Project URL from dashboard (e.g. `https://xxxx.supabase.co`). |
| `SUPABASE_SERVICE_KEY` or `SUPABASE_KEY` | Yes | Service role key from Project Settings → API. |
| `SUPABASE_BUCKET` | No | Bucket name (default: `eye-test-sessions`). |

Example **`.env`**:

```env
REMOTE_STORAGE=supabase
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_KEY=your-service-role-key
SUPABASE_BUCKET=eye-test-sessions
```

The API server loads `.env` automatically when started (using `python-dotenv`).

---

## 5. Install Python client

```bash
pip install supabase
```

---

## 6. Behavior

- When you **Sign off** (store session), the backend writes CSV and metadata locally and uploads to your bucket:
  - `{session_id}.csv` — session rows (19-column schema)
  - `{session_id}_metadata.json` — session metadata (prescription, AR/Lenso, quality metrics, etc.)
- If upload fails (wrong URL, key, or bucket), the error is logged and returned in the API; local files are still saved.

---

## Troubleshooting: files not appearing in bucket

### 403 Unauthorized / "new row violates row-level security policy"

This error means Supabase Storage is rejecting the upload because of **Row Level Security (RLS)**. In almost all cases the cause is:

**You are using the wrong API key.** The backend must use the **service_role** key (secret), not the **anon** (public) key. The anon key is subject to RLS; the service role bypasses it.

**Fix:**

1. In Supabase dashboard go to **Project Settings** (gear) → **API**.
2. Under **Project API keys** you will see:
   - **anon** / **public** — do **not** use this for `SUPABASE_SERVICE_KEY`.
   - **service_role** / **secret** — copy this value and use it for `SUPABASE_SERVICE_KEY` (or `SUPABASE_KEY`) in your `eye_test_engine/.env`.
3. Restart the API server after changing `.env`.

If you previously pasted the anon key, replace it with the service_role key and try Sign-off again. Local save will still work; only the cloud upload will start succeeding once the key is correct.

**If you’re sure the service_role key is in `.env` and the server was restarted** but you still get this error, add Storage policies in Supabase so uploads are allowed:

1. In Supabase dashboard go to **SQL Editor** and run the following (change `eye-test-sessions` if your bucket name is different):

```sql
-- Allow uploads (insert) to your bucket
create policy "Allow insert eye-test-sessions"
on storage.objects for insert to public
with check (bucket_id = 'eye-test-sessions');

-- Allow select (needed for upsert)
create policy "Allow select eye-test-sessions"
on storage.objects for select to public
using (bucket_id = 'eye-test-sessions');

-- Allow update (needed for upsert)
create policy "Allow update eye-test-sessions"
on storage.objects for update to public
using (bucket_id = 'eye-test-sessions')
with check (bucket_id = 'eye-test-sessions');
```

2. If you see “policy already exists”, either drop the existing policy first or use a different policy name.

After adding these policies, try Sign-off again. For production, prefer using the **service_role** key so the server bypasses RLS and you don’t rely on permissive policies.

---

- **"Bucket not found" (404)**  
  The bucket does not exist yet. In Supabase dashboard: **Storage** (left sidebar) → **New bucket** → enter the exact name (e.g. `eye-test-sessions`) → **Create bucket**. The name must match `SUPABASE_BUCKET` in your `.env` exactly (case-sensitive).

- **Use the service_role key**  
  In Supabase → Project Settings → API, use the **service_role** (secret) key for `SUPABASE_SERVICE_KEY` or `SUPABASE_KEY`. The **anon** (publishable) key does not have permission to upload to Storage in most setups; uploads will fail with 403 or policy errors.

- **Bucket must exist**  
  Create the bucket in the dashboard (Storage → New bucket). The name must match `SUPABASE_BUCKET` (default: `eye-test-sessions`).

- **Check the error message**  
  After Sign-off, if you see "Cloud save failed: …" in the UI, the message is the exact error from Supabase (e.g. invalid key, bucket not found, RLS policy). Also check server logs for `[REMOTE_STORAGE] Upload failed for <session_id>: <error>`.

---

## How to confirm data is saved in Supabase

1. **After Sign-off in the UI**  
   - **“Saved to supabase.”** — Upload succeeded; files are in your bucket.  
   - **“Cloud save failed: …”** — Local save worked but Supabase upload failed; check the error.

2. **Server logs**  
   - `[REMOTE_STORAGE] Uploaded session <id> to supabase` — success.  
   - `[REMOTE_STORAGE] Upload failed for <id>: <error>` — failure.

3. **Supabase dashboard**  
   Open **Storage** → your bucket; you should see `{session_id}.csv` and `{session_id}_metadata.json` after each stored session.

4. **Checklist**  
   - `REMOTE_STORAGE=supabase`  
   - `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` (or `SUPABASE_KEY`) set correctly.  
   - Bucket exists and name matches `SUPABASE_BUCKET` (default `eye-test-sessions`).  
   - You are using the **service_role** key, not the anon key.
