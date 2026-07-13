# Validation: Visitor Count

**Date**: 2026-07-12

## Automated Validation

- `task api:lint`: passed; all API source and tests passed Ruff checks.
- `AITRAF_REDIS_URL=redis://host.docker.internal:6380/0 task api:test`:
  passed; 38 tests completed, including five real-Redis integration tests.
- `docker compose config --quiet`: passed.
- `git diff --check`: passed.

Coverage includes strict Redis configuration, first and sequential increments,
100 concurrent increments, invalid stored values and types, unavailable Redis,
application lifecycle validation/closure, public access, empty-input enforcement,
explicit 500/503 failures, and unchanged authentication on the existing demo
route.

## Command-Level Smoke Validation

The endpoint was exercised with a single unauthenticated `POST` request and
returned `200` with the required `{"count": <integer>}` shape. Concurrency is
validated by the real-Redis integration test using an isolated key that is
deleted afterward, so validation does not add synthetic visits to the displayed
counter.

## Restart Persistence

The fixed key was set to `100`, the Redis container was restarted with
`docker compose restart redis`, and the value after restart remained `100`.
The validation key was deleted afterward. The named volume was not deleted.

## Repository Scope Audit

`git diff --name-only` and untracked-file inspection show only the repository's
local API/Redis `compose.yaml`, API package, tests, docs, lockfile, and feature artifacts.
No external or production deployment Compose, proxy, network, or volume
configuration was created or modified.

## Architecture And Input Audit

The visitor-count implementation contains:

- one fixed key: `aitraf:visitor-count:homepage`;
- one endpoint: `POST /visitor-count`;
- one atomic production mutation: Redis `INCR`;
- no process-local production count or fallback store;
- no visitor, session, network-address, cookie, or fingerprint collection;
- no alternate input shapes, keys, endpoints, aliases, normalization branches,
  deprecation paths, or compatibility scaffolding.

Production deployment remains an external handoff through the single required
`AITRAF_REDIS_URL` setting.
