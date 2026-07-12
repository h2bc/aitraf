# Data Model: Visitor Count

## Visitor Count

The single persistent aggregate for accepted homepage page views.

### Fields

- **identity**: Constant `aitraf:visitor-count:homepage`; not supplied by callers.
- **count**: Base-10 non-negative integer in Redis's supported signed 64-bit
  range.

### Storage Representation

- Redis string key: `aitraf:visitor-count:homepage`
- Redis value: canonical integer representation managed by `INCR`
- Missing key: valid only before the first increment; Redis `INCR` creates it
  with value `1`.
- Existing key: must be readable as an integer and must be non-negative before
  the application becomes ready.

### Validation Rules

- The identity is application-owned and immutable.
- The stored Redis type must be a string containing an integer.
- Negative, malformed, alternate-type, or out-of-range state is invalid and
  blocks readiness or fails the request if corruption appears after startup.
- The implementation never resets, repairs, decrements, expires, or deletes the
  count.

### State Transitions

```text
absent --first successful INCR--> 1
N      --successful INCR-------> N + 1
invalid --startup/request------> explicit failure, no repair
unavailable --startup/request--> explicit failure, no fallback
```

## Page Visit

One accepted invocation of the public recording operation.

### Fields

The request has no fields. Page identity, visitor identity, session identity,
network address, and idempotency identity are not part of the application data
model.

### Relationship

Each successful page visit produces exactly one atomic increment of the Visitor
Count and returns that operation's resulting count. A repeated or retried
invocation is a new page visit.

## Visitor Count Response

The representation returned after a successful increment.

### Fields

- **count**: Required non-negative integer equal to the value returned by the
  successful atomic increment.

### Validation Rules

- No additional fields are returned.
- A count is never synthesized from process memory or a stale read.
- A storage error produces an error response rather than a count response.

## Visitor Count Configuration

Required application configuration for the Redis boundary.

### Fields

- **redis_url**: Required Redis connection URL supplied through
  `AITRAF_REDIS_URL`.

### Validation Rules

- The variable must be present and non-empty.
- Connection failure, authentication failure, or inability to read and
  increment the configured Redis database is an explicit failure.
- Local host-run development uses the published local Redis port. Production
  receives a Redis URL from the separately owned deployment environment.

## Persistence Resources

- **Redis service**: Repository-managed Redis 7 runtime for local development;
  externally provisioned Redis-compatible runtime in production.
- **Named volume**: Owns `/data` and survives container replacement.
- **AOF configuration**: Append-only persistence with `appendfsync everysec`.
- **Snapshot configuration**: Periodic snapshot retained as an additional
  recovery artifact.

These resources preserve the count across normal restarts. They do not create
off-host backup or strict zero-loss crash durability.
