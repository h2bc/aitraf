# Contract: Visitor Count

## Record A Page View

```http
POST /visitor-count HTTP/1.1
Accept: application/json
```

### Authentication

None. The endpoint is intentionally public. The existing bearer token remains
required for existing protected endpoints and must not be sent by browser code
to this endpoint.

### Request

- No request body
- No query parameters
- No page, visitor, session, cookie, or idempotency identifier

Callers that send body fields or query parameters do not match the contract.

### Success Response

Status: `200 OK`

```json
{
  "count": 42
}
```

`count` is the required non-negative integer returned by the atomic increment.
No other response fields are present.

### Failure Responses

- `422 Unprocessable Entity`: request does not match the empty request contract.
- `503 Service Unavailable`: Redis is unavailable or refuses the operation.
- `500 Internal Server Error`: stored counter state is invalid or the increment
  returns a value outside the required representation.

Failures never include a guessed, cached, reset, or process-local count.

### Counting Semantics

- Every successful invocation increments the fixed homepage count once.
- Concurrent successful invocations do not lose increments.
- A retry is another invocation and may increment again.
- The endpoint does not assert that a page view represents a unique person.
- No visitor-level data is retained.

## Runtime Configuration Contract

```text
AITRAF_REDIS_URL=<required Redis URL>
```

Local host-run API:

```text
AITRAF_REDIS_URL=redis://localhost:6379/0
```

Example container address supplied by the external deployment environment:

```text
AITRAF_REDIS_URL=redis://redis:6379/0
```

Missing or invalid configuration prevents startup. Redis connectivity and the
existing stored value are validated before readiness.

## Infrastructure Contract

### Local Development

- Repository-root Compose owns a Redis 7 service.
- Redis publishes a configurable port for the API process running directly in
  the development environment.
- A named volume mounts `/data`.
- AOF `everysec`, periodic snapshotting, and a health check are enabled.

### External Production Runtime

- Production infrastructure is owned by a separate deployment repository and
  is not created or changed by this feature.
- That environment must supply a reachable persistent Redis service through
  `AITRAF_REDIS_URL` before the API can become ready.
- Network topology, volumes, health dependencies, persistence configuration,
  and backup policy are responsibilities of that external deployment.
