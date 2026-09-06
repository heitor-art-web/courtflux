# Supabase environment and branching policy

## Environment boundaries

CourtFlux production uses the production Supabase project only from production deployments and trusted production workers.

- Local development should use a local Supabase stack whenever database/auth changes are being developed.
- Vercel Preview or staging deployments must not use the production `SUPABASE_SERVICE_ROLE_KEY` or production worker credentials.
- A future Supabase preview/staging branch must have its own project URL, publishable key, privileged server credential, storage state, and test-only data.
- Never copy production user/auth data into development or preview environments.

## Migration-first rule

Once a schema is managed through migrations, do not make untracked schema changes directly in the hosted SQL/Table editors.

For every DDL change:

1. create or capture the migration SQL under `supabase/migrations/`;
2. test the migration against a fresh/local database where practical;
3. review RLS, grants, views, triggers and `SECURITY DEFINER` functions;
4. apply the migration to the intended Supabase environment;
5. run Supabase security/performance advisors;
6. commit the exact migration file so Git and the remote migration history stay aligned.

## Current migration repair note

The Supabase branch metadata currently reports `MIGRATIONS_FAILED` even though the production project is healthy. Known recent security migrations have been synchronized into this repository. Historical migration timestamps still need to be reconciled against the exact failed branch replay log before any migration-history repair is performed.

Do not rewrite `supabase_migrations.schema_migrations` manually. Use the Supabase branching workflow logs plus `supabase migration fetch/list/repair` to reconcile history, then validate by creating/rebasing a non-production branch.

## Staging requirement

Before treating a Vercel Preview deployment as a real staging environment, provision a separate Supabase branch/project and configure Preview-only credentials. Until then, Preview deployments must be treated as application previews rather than isolated database staging.
