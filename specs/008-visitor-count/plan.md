# Implementation Plan: Visitor Count

**Branch**: `[008-visitor-count]` | **Date**: 2026-07-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/008-visitor-count/spec.md`

## Summary

Add one unauthenticated API endpoint that atomically increments and returns a
single lifetime homepage page-view count stored in a dedicated Redis instance.
Keep Redis access in an API-owned visitor-count feature, require and validate
`AITRAF_REDIS_URL`, connect during application creation so invalid or
unavailable storage prevents readiness, and use Redis `INCR` as the only
increment operation. Add a root local-development Compose service with a named
volume and AOF persistence. Production deployment and the frontend are owned by
other repositories; this repository exposes only the required Redis URL and HTTP
contracts for those external consumers.

## Technical Context

**Language/Version**: Python `>=3.10,<3.14`

**Primary Dependencies**: FastAPI, Pydantic, python-dotenv, redis-py with its
async client, and Redis 7 Alpine for local infrastructure

**Storage**: One dedicated Redis instance; fixed key
`aitraf:visitor-count:homepage`; integer value incremented only with atomic
`INCR`; named volume plus append-only persistence using `appendfsync everysec`

**Testing**: Pytest API unit/route tests with an explicit fake counter boundary,
real-Redis integration tests for concurrent increments and restart persistence,
Compose configuration validation, and command-level HTTP smoke validation

**Target Platform**: Linux API container/runtime with a required Redis URL;
local Docker Compose dependency for host-run development

**Project Type**: Python monorepo with an independently installable FastAPI
package and an externally deployed Next.js frontend

**Performance Goals**: At least 95% of visit increments complete within one
second under normal demo traffic; 100 concurrent accepted increments produce
exactly 100 count increases with no lost updates

**Constraints**: The endpoint is public and must not use or expose
`AITRAF_API_TOKEN`; one empty request shape and one fixed counter key; no visitor
identity, uniqueness detection, bot filtering, rate limiting, event history, or
in-memory fallback; Redis unavailability fails explicitly; AOF `everysec`
provides normal-restart persistence but not zero-loss durability for abrupt
host failure; network retries are separate page views

**Scale/Scope**: One lifetime homepage counter, one API endpoint, one Redis key,
API configuration/tests/docs, and a local Redis Compose service; production
deployment, frontend implementation, and analytics remain outside scope

## Constitution Check

*GATE: Passed before Phase 0 and re-checked after Phase 1 design.*

- **No Excessive Fallbacks**: PASS. `AITRAF_REDIS_URL` is required, startup
  verifies Redis connectivity and stored value validity, and request-time Redis
  failures surface explicitly. There is no in-memory or reset-to-zero fallback.
- **Package By Feature**: PASS. Route, schema, service, and Redis boundary stay
  under `packages/aitraf-api/.../features/visitor_count`. No counter behavior is
  promoted to core or added to training/ML packages.
- **Function Decomposition**: PASS. Configuration loading, Redis connection,
  stored-state validation, atomic increment, response mapping, and routing have
  focused boundaries.
- **Functional Programming And State**: PASS. The fixed identity and response
  mapping are immutable. The Redis client is framework-managed application
  state localized at application startup/shutdown and passed explicitly to the
  service boundary.
- **Reproducibility**: PASS. The Redis image, command, persistence settings,
  volume, URL, key, local startup command, test commands, restart validation,
  and concurrency smoke test are specified.
- **No Legacy Compatibility**: PASS. This is a new capability with one contract;
  no aliases, alternate endpoints, old counter paths, or dual stores are added.
- **Required Types Over Defensive Normalization**: PASS. The endpoint accepts no
  body, query, identity, or alternate page input. Redis must contain either no
  key before first increment or one valid non-negative base-10 integer; other
  stored types and values fail.

## Project Structure

### Documentation (this feature)

```text
specs/008-visitor-count/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── visitor-count.md
└── tasks.md              # generated later by /speckit-tasks
```

### Source Code (repository root)

```text
packages/aitraf-api/
├── pyproject.toml
├── README.md
├── src/aitraf_api/
│   ├── app.py
│   ├── config.py
│   └── features/
│       ├── __init__.py
│       └── visitor_count/
│           ├── __init__.py
│           ├── route.py
│           ├── schemas.py
│           └── service.py
└── tests/
    ├── test_app.py
    ├── test_config.py
    └── features/visitor_count/
        ├── conftest.py
        ├── test_route.py
        └── test_service.py

compose.yaml               # local Redis dependency only
.env.example
README.md
```

**Structure Decision**: Add a dedicated `visitor_count` feature to
`aitraf-api`. The service module owns the narrow Redis protocol and Redis-backed
implementation because there is only one consumer; no shared storage abstraction
is added to `aitraf-core`. Application creation owns client construction,
connectivity/state validation, and lifecycle cleanup. The root `compose.yaml`
owns local application dependencies and initially contains only Redis so the API
can continue running directly through the existing task command. The separate
production deployment repository is not modified or planned as work here.

## Implementation Direction

1. Add redis-py to `aitraf-api` and require exactly one `AITRAF_REDIS_URL` string
   in strict API settings, `.env.example`, tests, and documentation.
2. Add an API-owned counter protocol and Redis implementation whose production
   increment is one `INCR` call and whose result must be a non-negative integer.
3. During application creation, build the Redis client, ping it, validate that
   an existing fixed key is a non-negative integer, and attach the client-backed
   counter to application state; close it through the application lifespan.
4. Add `POST /visitor-count` with no authentication, body, query parameters, or
   caller-controlled page identity; return only `{"count": <integer>}`.
5. Keep existing demo routes protected by the bearer token and avoid embedding
   that token in frontend browser code. Browser-origin policy is deployment
   configuration rather than visitor identity or authentication.
6. Add root `compose.yaml` with a non-public Redis 7 service exposed on a
   configurable host port for host-run development, AOF `everysec`, a named
   volume, health check, and no API container duplication.
7. Document the external runtime requirement only: deployments must provide a
   reachable persistent Redis instance through `AITRAF_REDIS_URL`. Do not add or
   modify production deployment files in this repository.
8. Use focused fake-backed unit/route tests and real Redis integration coverage
   for atomic concurrency, malformed stored state, unavailable Redis, and
   persistence across a normal restart.
9. Validate local Compose, start Redis, run package tests, exercise sequential
   and 100-way concurrent HTTP increments, restart Redis without deleting the
   volume, and confirm the count remains.

## Post-Design Constitution Check

- **No Excessive Fallbacks**: PASS. Contracts require Redis and explicit failure
  for missing configuration, invalid stored state, or unavailable storage.
- **Package By Feature**: PASS. All runtime logic remains inside the API feature;
  Compose is infrastructure configuration, not a parallel application layer.
- **Function Decomposition**: PASS. External state operations are reduced to
  connectivity validation, stored-value validation, and one atomic increment.
- **Functional Programming And State**: PASS. Mutable connection state is
  lifecycle-bound and no process-local count exists.
- **Reproducibility**: PASS. Design artifacts define exact key, image,
  persistence mode, URLs, commands, and expected validation results.
- **No Legacy Compatibility**: PASS. One new endpoint and one Redis configuration
  path are introduced without compatibility surfaces.
- **Required Types Over Defensive Normalization**: PASS. The HTTP and Redis
  contracts each define one exact representation and reject alternatives.

## Complexity Tracking

No constitution violations require justification.
