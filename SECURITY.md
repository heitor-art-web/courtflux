# Security Operations

## Privileged credentials

- Keep `SUPABASE_SERVICE_ROLE_KEY` and all other privileged credentials server-side only.
- Never expose privileged credentials through `NEXT_PUBLIC_*`, client bundles, logs, screenshots, issue bodies, or analytics payloads.
- Frontend Supabase access must use only the publishable key and remain protected by RLS and least-privilege grants.

## Secret rotation policy

Rotate privileged credentials immediately when any of these occur:

- suspected or confirmed exposure;
- a collaborator with production access leaves or no longer needs access;
- a secret is committed, pasted into a ticket/chat, or logged;
- a provider reports credential compromise;
- a production environment or deployment boundary is rebuilt after an incident.

Also perform a scheduled credential review at least quarterly and rotate long-lived privileged credentials when operationally safe.

### Rotation procedure

1. Create the replacement credential without revoking the current credential first.
2. Update the trusted server/deployment environment only.
3. Deploy and verify Supabase health plus the privileged server workflows that use the credential.
4. Revoke or disable the previous credential only after the replacement is confirmed working.
5. Re-run security checks and inspect deployment/application logs for authentication failures.
6. Record the rotation date and affected systems without storing the credential value.

If exposure is suspected, treat rotation as an incident: revoke compromised credentials as soon as continuity allows, invalidate affected sessions where relevant, and review logs for misuse.

## Administrative access

- Require MFA on every Supabase Owner or Administrator account.
- Keep the number of Owners/Administrators to the minimum operationally necessary.
- Remove dormant collaborators promptly.
- Do not use shared administrator accounts.

## Logging

Never log passwords, access tokens, refresh tokens, JWTs, service-role/secret keys, database passwords, Authorization headers, or complete environment objects.

## Index maintenance

Supabase `unused_index` findings are informational and are not sufficient by themselves to justify dropping an index.

Before removing an index:

1. confirm it does not back a primary key, unique constraint, or other required constraint;
2. confirm it is not needed for a foreign-key/join/filter path expected in production;
3. review meaningful production usage over time rather than a newly created database's zero-scan count;
4. compare it for exact or prefix redundancy with existing indexes;
5. make the change through a reversible migration and rerun Supabase performance advisors.

For this early-stage project, zero-scan indexes should be reviewed again after meaningful production traffic rather than removed pre-emptively.
