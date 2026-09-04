# Basketball AI Analytics — Vercel Beta

This directory is the Vercel-facing control plane for the project. It intentionally does **not** run YOLO, PyTorch, OpenCV, yt-dlp, or the basketball pipeline inside Vercel Functions.

## Architecture

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

The worker can initially run on the development Mac. Later it can move to a GPU server without changing the browser application; only `WORKER_API_URL` changes.

## 1. Start the Python worker

From the repository root on the machine that has the model weights:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export WORKER_API_TOKEN="replace-with-a-long-random-secret"
export WORKER_HOST="127.0.0.1"
export WORKER_PORT="8787"
python -m services.worker_api
```

Check it locally:

```bash
curl http://127.0.0.1:8787/health \
  -H "Authorization: Bearer $WORKER_API_TOKEN"
```

The worker defaults to one analysis at a time (`WORKER_MAX_JOBS=1`) to avoid multiple MPS jobs competing for memory.

## 2. Give Vercel an HTTPS worker URL

Vercel cannot call `localhost` on your Mac. The worker must be reachable through a public **HTTPS** URL. For a private beta, expose port 8787 through a tunnel/reverse proxy that you control (for example a Cloudflare Tunnel or another authenticated HTTPS tunnel).

Keep `WORKER_API_TOKEN` enabled even when the tunnel itself has access controls.

## 3. Run the Next.js app locally

```bash
cd web
cp .env.example .env.local
# Edit WORKER_API_URL and WORKER_API_TOKEN
npm install
npm run dev
```

Open http://localhost:3000.

## 4. Deploy to Vercel

Import the GitHub repository in Vercel and configure:

- **Framework Preset:** Next.js
- **Root Directory:** `web`
- **Build Command:** default (`next build`)
- **Environment Variables:**
  - `WORKER_API_URL` — public HTTPS URL of the Python worker
  - `WORKER_API_TOKEN` — same token configured on the worker

Do not prefix either variable with `NEXT_PUBLIC_`; they are server-only secrets.

Every push can then produce a Vercel preview deployment. Vercel has first-class Next.js support, so no Python ML dependencies are included in the Vercel deployment.

## Job behavior

The browser submits:

- video source
- start/end seconds
- worker-local model path
- detector confidence
- inference image size
- device and tracker
- complete `match_config.json`

The worker creates an isolated job directory under `evaluation/worker_jobs/<job_id>/`, saves the exact match config and request, starts the existing analyzer, and exposes status/summary/logs back to Vercel.

The analyzer now accepts `--match-config`, so remote jobs do not overwrite the repository-wide `data/match_config.json`.

## Security notes

- The worker API requires a bearer token.
- Vercel keeps the token server-side; the browser only calls `/api/...` on the Vercel deployment.
- Do not expose the worker without HTTPS and authentication.
- Keep `WORKER_MAX_JOBS=1` while using the development Mac/MPS.
- The current worker is a beta execution service, not a multi-user production queue.
