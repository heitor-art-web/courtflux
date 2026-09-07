# Production worker backlog

This file tracks worker/network hardening that is intentionally deferred while CourtFlux is developed and validated locally.

## Before production launch

- [ ] Decide the production worker topology (private network, managed GPU worker, or public HTTPS service).
- [ ] Remove reliance on the local loopback fallback for production deployments and configure an explicit `WORKER_API_URL`.
- [ ] Require service-to-service authentication when the worker is reachable beyond loopback. `WORKER_API_TOKEN` is the current simple mechanism; reassess whether platform-native identity, signed requests, mTLS, or another private-channel mechanism is preferable.
- [ ] Ensure the worker refuses unauthenticated requests on non-loopback binds. Keep this as a deployment/security check even if the authentication mechanism changes.
- [ ] Keep browser/user authentication separate from backend-to-worker trust. Users authenticate to CourtFlux; browsers must never receive worker credentials.
- [ ] Add rate limits / queue limits appropriate to the production compute environment and verify that customer usage/entitlement checks cannot be bypassed by calling the worker directly.
- [ ] Verify HTTPS, secrets storage/rotation, logging, request attribution, and incident-revocation procedure.
- [ ] Add production deployment documentation and a smoke test covering web backend -> worker health -> job submission -> status/log polling.

## Current local-development decision

For local Beta Lab development, no worker URL or token is required:

- Next.js defaults to `http://127.0.0.1:8787` when `WORKER_API_URL` is unset.
- The Python worker defaults to `WORKER_HOST=127.0.0.1` and permits tokenless requests only while configured for a loopback bind.
- Setting `WORKER_API_TOKEN` restores bearer-token enforcement even on localhost.
- Configuring a non-loopback `WORKER_HOST` without a token is rejected.

This local convenience must not be treated as the final production security design.
