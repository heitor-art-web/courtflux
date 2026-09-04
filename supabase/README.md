# CourtFlux Supabase architecture

Supabase is the platform source of truth for identity, workspaces, subscriptions, analysis jobs, run summaries, metrics and usage. The Python vision pipeline keeps local/batched persistence for frame-hot-path work so network latency never blocks inference.

## Identity and tenancy

- `auth.users`: Supabase Auth identity.
- `profiles`: user-facing profile metadata.
- `workspaces`: billing/data isolation boundary. Every new user automatically receives a personal workspace.
- `workspace_members`: roles are `owner`, `admin`, `analyst`, `viewer`.

The `auth.users` signup trigger creates the profile, personal workspace, owner membership and a current Free subscription in one transaction.

## Billing and subscriptions

- `plans`: Free, Base and Premium product tiers plus feature/limit JSON.
- `plan_prices`: provider-neutral monthly/yearly prices and external price IDs.
- `billing_customers`: maps a workspace to a billing provider customer.
- `subscriptions`: current subscription state and provider subscription ID.
- `subscription_events`: append-only webhook/audit events.
- `usage_events`: metered usage such as `analysis_minutes`.
- `workspace_entitlements`: security-invoker view exposing the current plan, features and limits to workspace members.

Pricing and quotas are intentionally not hard-coded yet. A payment provider such as Stripe can be attached later without changing the account or analysis schema.

## Basketball and analysis data

- `teams`, `games`: workspace basketball data.
- `model_versions`: versioned CourtFlux model metadata/evaluation results.
- `analysis_runs`: durable run record, parameters and final summary JSON.
- `analysis_jobs`: queue/worker lifecycle linked 1:1 to a run.
- `analysis_metrics`: queryable numeric run metrics (`ball.*`, performance metrics, etc.).
- `analysis_artifacts`: pointers to future Storage objects such as `summary.json`, `frames.jsonl`, logs and previews.

The worker writes lifecycle updates and completed summaries with the Supabase service-role key. Numeric summary fields are flattened into `analysis_metrics`, making cross-video P0 regression queries possible without parsing JSON.

## RLS model

All application tables use Row Level Security. Users can read only workspaces they belong to. Team/game editing is role-based. Billing writes, run status updates, metrics and usage are server/worker operations using the service-role key. Internal RLS helper and trigger functions live in the non-exposed `private` schema.

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
  -> Supabase: analysis_runs + analysis_jobs (queued)
  -> Python worker using the same platform job UUID
  -> Supabase: running/completed/failed
  -> Supabase: summary + flattened metrics + usage
  -> Browser polls Supabase-backed /api/jobs/:id
```

Worker-local `job.json`, `console.log` and SQLite remain operational scratch/debug data. They are no longer the intended platform source of truth.
