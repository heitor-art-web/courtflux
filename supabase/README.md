# CourtFlux Supabase architecture

Supabase is the platform source of truth for identity, workspaces, subscriptions, analysis jobs, run summaries, metrics and usage. The Python vision pipeline keeps local/batched persistence for frame-hot-path work so network latency never blocks inference.

## Identity and tenancy

- `auth.users`: Supabase Auth identity.
- `profiles`: user-facing profile metadata.
- `workspaces`: billing/data isolation boundary. Every new user automatically receives a personal workspace.
- `workspace_members`: roles are `owner`, `admin`, `analyst`, `viewer`.

The `auth.users` signup trigger creates the profile, personal workspace, owner membership and a current Beta subscription in one transaction.

## Plans, usage and subscriptions

The public product plans are:

- `beta`: 60 analysis minutes/month.
- `coach`: 300 analysis minutes/month.
- `team`: 720 analysis minutes/month.
- `academy`: 2,400 analysis minutes/month, with the public package positioned for up to five teams.

Plan limits are stored in `plans.limits` under `analysis_minutes_monthly`. The private-beta usage policy warns at 80% and pauses new customer analyses at 100%. Billing and automatic overage charges remain disabled during private beta.

- `plans`: Beta, Coach, Team and Academy product tiers plus feature/limit JSON.
- `plan_prices`: provider-neutral monthly/yearly prices and external price IDs for later billing activation.
- `billing_customers`: maps a workspace to a billing provider customer.
- `subscriptions`: current subscription state and provider subscription ID.
- `subscription_events`: append-only webhook/audit events.
- `usage_events`: metered usage such as `analysis_minutes`.
- `workspace_usage_monthly`: security-invoker monthly usage rollup.
- `usage_allowance_grants`: trusted-admin top-ups that add temporary analysis minutes without changing the plan or charging an overage.
- `workspace_entitlements`: security-invoker view exposing the current plan, features and limits to workspace members.

Authenticated members can read their allowance grants through RLS, but grants have no authenticated insert/update/delete policy. Trusted server/service-role operations are the intended way to award additional private-beta minutes.

A payment provider such as Stripe can be attached later without changing the account, usage or analysis schema. Public prices remain product packaging until billing is explicitly activated.

## Basketball and analysis data

- `teams`, `games`: workspace basketball data.
- `model_versions`: versioned CourtFlux model metadata/evaluation results.
- `analysis_runs`: durable run record, parameters and final summary JSON.
- `analysis_jobs`: queue/worker lifecycle linked 1:1 to a run.
- `analysis_metrics`: queryable numeric run metrics (`ball.*`, performance metrics, etc.).
- `analysis_artifacts`: pointers to future Storage objects such as `summary.json`, `frames.jsonl`, logs and previews.

The worker writes lifecycle updates and completed summaries with the Supabase service-role key. Numeric summary fields are flattened into `analysis_metrics`, making cross-video P0 regression queries possible without parsing JSON. Completed processed-video duration is written to `usage_events` as `analysis_minutes`.

## RLS model

All application tables use Row Level Security. Users can read only workspaces they belong to. Team/game editing is role-based. Billing writes, allowance-grant writes, run status updates, metrics and usage are server/worker operations using the service-role key. Internal RLS helper and trigger functions live in the non-exposed `private` schema.

## Environment

Web/Vercel:

```text
NEXT_PUBLIC_SUPABASE_URL=https://pmpujloaixzztnrqnhbx.supabase.co
NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY=<publishable key>
SUPABASE_SERVICE_ROLE_KEY=<server-only service-role key>
WORKER_API_URL=<worker HTTPS URL>
WORKER_API_TOKEN=<shared worker bearer token>
```

Python worker:

```text
SUPABASE_URL=https://pmpujloaixzztnrqnhbx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<server-only service-role key>
WORKER_API_TOKEN=<shared worker bearer token>
```

Never expose or commit the service-role key.

## Request flow

```text
Authenticated browser
  -> Next.js /api/jobs
  -> read entitlement + current monthly usage + active top-ups
  -> if allowance is exhausted: return a soft-stop response (no charge)
  -> otherwise Supabase: analysis_runs + analysis_jobs (queued)
  -> Python worker using the same platform job UUID
  -> Supabase: running/completed/failed
  -> Supabase: summary + flattened metrics + usage
  -> Browser polls Supabase-backed /api/jobs/:id
```

Worker-local `job.json`, `console.log` and SQLite remain operational scratch/debug data. They are no longer the intended platform source of truth.
