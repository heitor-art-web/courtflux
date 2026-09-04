# CourtFlux Web

This directory contains the Next.js/Vercel web application for CourtFlux.

The web app now has two distinct responsibilities:

- `/` is the public, crawlable CourtFlux marketing and SEO surface.
- `/lab` is the internal/private-beta analysis control plane that talks to the Python worker.

Authentication and account routes live alongside them but are explicitly excluded from search indexing.

## Route architecture

```text
/          Public CourtFlux marketing site
/login     Sign in / create account (noindex)
/account   Authenticated workspace and entitlement view (noindex)
/lab       Beta Lab / engineering analysis interface (noindex)
/api/*     Server-side worker proxy endpoints
```

The public site is server-rendered with Next.js metadata, canonical URL support, Open Graph metadata, `robots.txt`, `sitemap.xml`, structured data and optimized image delivery.

## Analysis architecture

```text
Browser
  |
  v
Vercel / Next.js (`web/`)
  |
  | HTTPS + Bearer token (server-side only)
  v
Python worker (`services/worker_api.py`)
  |
  v
Existing analyzer / YOLO / MPS / yt-dlp
  |
  v
Run summary + logs
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
- `SUPABASE_SERVICE_ROLE_KEY` — server-only Supabase admin key. Never expose as `NEXT_PUBLIC_*`.

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

## Search/indexing rules

Only public marketing content should be indexable. Login, account, Beta Lab and API surfaces use explicit noindex or crawl exclusions as appropriate.

When the production domain is ready, set `NEXT_PUBLIC_SITE_URL` to the canonical domain and verify the live `robots.txt`, `sitemap.xml`, canonical tags and social previews.

## Security notes

- The worker API requires a bearer token.
- Vercel keeps the worker token server-side; the browser only calls `/api/...` on the Vercel deployment.
- Do not expose the worker without HTTPS and authentication.
- Keep `WORKER_MAX_JOBS=1` while using the development Mac/MPS.
- The current worker is a beta execution service, not a multi-user production queue.
