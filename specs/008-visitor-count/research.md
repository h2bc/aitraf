# Research: Visitor Count

## Dedicated Redis Counter

**Decision**: Use a dedicated AITRAF Redis 7 instance and increment one fixed key
with the atomic `INCR` command.

**Rationale**: The feature needs one concurrent counter, and `INCR` performs the
increment and returns the resulting integer as one atomic operation. This avoids
lost updates without introducing visitor-event rows or read-modify-write logic.
A dedicated instance keeps AITRAF deployment, data lifecycle, resource limits,
and failure handling independent from the existing Medusa Redis service.

**Alternatives considered**:

- Sharing the Medusa Redis instance or one of its logical databases was rejected
  because it couples AITRAF to another application's private network, lifecycle,
  backups, resource pressure, and administrative operations.
- PostgreSQL was rejected because AITRAF does not currently operate it and the
  feature needs only one atomic value rather than relational history.
- S3-compatible storage was rejected because object read-modify-write is not a
  suitable atomic counter primitive.
- Process memory was rejected because it resets and diverges across replicas.

## Persistence And Durability Boundary

**Decision**: Store Redis data on a named volume and enable AOF with
`appendfsync everysec`; retain periodic snapshots as an additional recovery aid.
Guarantee preservation across normal API/container restarts, while explicitly
accepting that an abrupt Redis or host failure can lose the most recent writes.

**Rationale**: AOF every second is a proportionate durability/performance choice
for a non-critical demo page-view total. A named volume survives container
replacement. The feature must not describe an acknowledged Redis increment as
financial-grade exactly-once durable storage.

**Alternatives considered**:

- Snapshot-only persistence was rejected because it can lose a longer interval
  of recently recorded visits.
- `appendfsync always` was rejected because a synchronous disk flush for every
  page view is disproportionate for an informational counter.
- No persistence was rejected because the count must survive normal restarts.
- Replication and off-host backups were deferred because host-loss recovery is
  outside this demo feature; a named volume is not represented as a backup.

## Page-View And Retry Semantics

**Decision**: Count each successful endpoint invocation as one page view. Do not
deduplicate, attach an idempotency token, or identify a visitor. Treat a retry as
a separate page view.

**Rationale**: This matches the requested browser-on-load behavior and avoids
cookies, fingerprints, IP storage, and session state. Atomicity prevents lost
concurrent increments, but it does not make an ambiguous network retry exactly
once; explicitly accepting retry overcount is simpler and honest.

**Alternatives considered**:

- Client session identifiers were rejected because the user requested a simple
  load call without visitor checks.
- Idempotency keys were rejected because they require client state and retention
  of request identities.
- Unique-person analytics were rejected as a different privacy and product
  feature.

## HTTP Contract

**Decision**: Expose `POST /visitor-count` without bearer authentication, body,
query parameters, or caller-selected page identity. Return status 200 with one
strict JSON object containing the resulting non-negative integer count.

**Rationale**: `POST` represents mutation and prevents crawlers, link previews,
or ordinary cacheable reads from incrementing through a `GET`. A fixed server
key meets the one-page scope and prevents arbitrary key creation. The existing
API token cannot be safely exposed to browser JavaScript.

**Alternatives considered**:

- An authenticated endpoint was rejected because a browser-held shared token is
  public in practice.
- `GET` with increment side effects was rejected because reads should remain
  safe and cache behavior could make results misleading.
- A request body containing page or visitor data was rejected because there is
  only one counter and alternate shapes add no value.

## Redis Client Ownership And Lifecycle

**Decision**: Use redis-py's async client. Construct and validate it at the
FastAPI application lifecycle boundary, store a narrow counter dependency on
application state, and close the client at shutdown. Fail readiness when Redis
cannot be reached or its existing key is invalid.

**Rationale**: The endpoint remains non-blocking, the network client is reused
through its connection pool, and external mutable state stays at the framework
boundary. Startup validation follows the repository's fail-loudly rules instead
of discovering missing configuration only after a public visit.

**Alternatives considered**:

- Creating a Redis client per request was rejected as unnecessary connection
  churn.
- A global module client was rejected as hidden state that complicates testing
  and shutdown.
- Continuing readiness with an unavailable counter was rejected as a silent
  degradation path.

## Local Dependency And External Runtime Boundary

**Decision**: Add a repository-root local Compose stack containing both the API
and Redis. Mount API/core sources and enable API reload for development. Treat
production Redis provisioning as an external runtime prerequisite owned by the
separate deployment repository.

**Rationale**: Compose natively owns both local service lifecycles, so one
foreground command and `Ctrl+C` start and stop the stack. Source mounts retain a
fast development loop. This repository defines its required connection contract
but does not own or change infrastructure maintained in another repository.

**Alternatives considered**:

- Installing Redis inside the devcontainer was rejected because an application
  dependency should have its own lifecycle and persistent volume.
- Production-only Redis was rejected because local development and integration
  validation would not reproduce required startup behavior.
- Running the API in the devcontainer while Redis runs detached was rejected
  because it splits local lifecycle ownership and requires custom cleanup.
