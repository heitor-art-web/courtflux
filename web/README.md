# CourtFlux Web

This directory contains the canonical Next.js/Vercel web application for CourtFlux.

The product has four clearly separated surfaces:

- `/` and the public landing pages are the crawlable CourtFlux marketing/SEO website.
- `/account` is the authenticated customer dashboard backed by Supabase workspace, entitlement, usage and analysis data.
- `/analysis/new` is the authenticated customer New Analysis flow for video upload/YouTube, match context and analysis-minute review.
- `/lab` is the internal/private-beta engineering control plane that talks directly to the Python worker.

Authentication and product routes live alongside the public site but are explicitly excluded from search indexing.

The old Streamlit `ui/beta.py` entrypoint is Beta Lab only. It no longer serves public marketing, signup or account pages. Those customer-facing flows belong to this Next.js app.

## Route architecture

```text
/                            Public CourtFlux marketing site
/basketball-video-analysis   Public search landing page
/login                       Sign in / create account (noindex)
/account                     Authenticated customer dashboard (noindex)
/analysis/new                Authenticated New Analysis flow (noindex)
/lab                         Beta Lab / engineering analysis interface (noindex)
/api/*                       Server-side worker/upload proxy endpoints
```

The public site is server-rendered with Next.js metadata, canonical URL support, Open Graph metadata, `robots.txt`, `sitemap.xml`, structured data and optimized image delivery.

## Customer analysis flow

`/analysis/new` is the customer-facing entry point for processing basketball footage. It deliberately keeps the engineering controls out of the customer journey.

The flow supports:

- private video upload through a short-lived signed Supabase Storage upload token,
- YouTube source URLs,
- required home/away team names,
- optional date, competition and active kit colours,
- optional jersey-number/player-name context without requiring a roster,
- editable analysis start/end times,
- an analysis-minute estimate before submission,
- warning/blocked states when the requested window would exceed the remaining monthly allowance,
- durable `games`, `analysis_runs` and `analysis_jobs` records in Supabase.

Uploaded videos are stored in the private `analysis-videos` bucket. The browser never receives the Supabase service-role key. The worker resolves `supabase://analysis-videos/...` sources with its own service credentials and streams them into a worker-local cache before inference.

## Customer dashboard

`/account` reads the signed-in user's Supabase workspace and presents customer-facing operational context rather than raw entitlement JSON:

- current workspace and role,
- current plan state,
- monthly `analysis_minutes` usage,
- remaining minutes and any beta top-up,
- total and active analyses,
- recent analysis states,
- current team context,
- direct access to `/analysis/new` for a new customer analysis.

Billing remains inactive during private beta. Usage shown in the dashboard comes from the `workspace_usage_monthly` read model, which rolls up worker-written `usage_events`.

## Analysis architecture

```text
Browser
  |
  | signed upload token (uploaded files only)
  v
Supabase private Storage

Browser
  |
  v
Vercel / Next.js (`web/`)
  |
  | HTTPS + Bearer token (server-side only)
  v
Python worker (`services/worker_api.py`)
  |
  | resolves YouTube or private Supabase video source
  v
Existing analyzer / YOLO / MPS/CUDA / yt-dlp
  |
  v
Supabase run state + summary + usage
```

The Python worker can initially run on the development Mac and later move to a GPU server without changing the browser application; only `WORKER_API_URL` changes.

## Environment variables

Copy the example file:

```bash
cd web
cp .env.example .env.local
```

Configure:

- `NEXT_PUBLIC_SITE_URL` — canonical production URL used by metadata, sitemap and social links.
- `WORKER_API_URL` — public HTTPS URL of the Python analysis worker.
- `WORKER_API_TOKEN` — bearer token shared with the worker. Keep server-side only.
- `NEXT_PUBLIC_SUPABASE_URL` — Supabase project URL.
- `NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY` — browser-safe Supabase publishable key.
- `SUPABASE_SERVICE_ROLE_KEY` — server-only Supabase admin key. The Python worker also needs it to resolve private uploaded videos. Never expose it as `NEXT_PUBLIC_*`.

## Run locally

From the repository root:

```bash
npm install
npm run dev
```

Or from this directory:

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

## Quality checks

From the repository root:

```bash
npm run typecheck
npm run lint
npm run build
```

The `Web quality` GitHub Actions workflow runs the same checks for frontend changes.

## Start the Python worker

From the repository root on the machine that has the model weights:

```bash
source .venv/bin/activate

export WORKER_API_TOKEN="replace-with-a-long-random-secret"
export WORKER_HOST="127.0.0.1"
export WORKER_PORT="8787"
python3 -m services.worker_api
```

Check it locally:

```bash
curl http://127.0.0.1:8787/health \
  -H "Authorization: Bearer $WORKER_API_TOKEN"
```

The worker defaults to one analysis at a time (`WORKER_MAX_JOBS=1`) to avoid multiple MPS jobs competing for memory.

## Give Vercel an HTTPS worker URL

Vercel cannot call `localhost` on the development Mac. The worker must be reachable through a public HTTPS URL. For a private beta, expose port 8787 through a tunnel or reverse proxy that you control.

Keep `WORKER_API_TOKEN` enabled even when the tunnel itself has access controls.

## Deploy to Vercel

Recommended project configuration:

- **Framework Preset:** Next.js
- **Root Directory:** `web`
- **Build Command:** default (`next build`)
- **Environment Variables:** the values described above

Do not prefix `WORKER_API_URL`, `WORKER_API_TOKEN` or `SUPABASE_SERVICE_ROLE_KEY` with `NEXT_PUBLIC_`.

Development branches prefixed with `assistant-` have automatic Vercel deployments disabled in repository configuration. Completed batches are merged to `main`, which is the deployment point.

## Search/indexing rules

Only public marketing content should be indexable. Login, account, New Analysis, Beta Lab and API surfaces use explicit noindex or crawl exclusions as appropriate.

When the production domain is ready, set `NEXT_PUBLIC_SITE_URL` to the canonical domain and verify the live `robots.txt`, `sitemap.xml`, canonical tags and social previews.

## Security notes

- The worker API requires a bearer token.
- Vercel keeps the worker token server-side; the browser only calls `/api/...` on the Vercel deployment.
- Uploaded source videos remain private; the browser receives only a short-lived signed upload token.
- The worker validates that an uploaded storage path belongs to the workspace that requested the analysis.
- Do not expose the worker without HTTPS and authentication.
- Keep `WORKER_MAX_JOBS=1` while using the development Mac/MPS.
- The current worker is a beta execution service, not a multi-user production queue.
