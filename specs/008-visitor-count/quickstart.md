# Quickstart: Validate Visitor Count

## Prerequisites

- Docker with Compose support
- Repository dependencies installed through the existing `uv` workflow
- Existing required API/MLflow/S3 environment variables
- `curl`

## Configure Local Redis

Set the required API connection value:

```bash
export AITRAF_REDIS_URL=redis://host.docker.internal:6380/0
```

Use `redis://localhost:6380/0` instead when the API runs directly on the Docker
host rather than in the repository devcontainer.

Build and start the repository-managed local stack in the foreground:

```bash
task api:run
```

The API waits for Redis health before starting. Press `Ctrl+C` to stop both
services. Validate the Compose configuration with:

```bash
docker compose config --quiet
```

## Run Automated Tests

Run API unit and route tests:

```bash
uv run --package aitraf-api pytest packages/aitraf-api/tests
```

Run the Redis integration tests with the local service available:

```bash
AITRAF_REDIS_URL="$AITRAF_REDIS_URL" \
  uv run --package aitraf-api pytest -m redis_integration
```

Expected results include successful first/sequential increments, public route
access, explicit configuration/storage failures, exact concurrent increments,
and valid stored-state enforcement.

## Start The API

Export the existing required API settings and run:

```bash
task api:run
```

Confirm readiness:

```bash
curl --fail http://127.0.0.1:8001/health
```

## Validate The HTTP Contract

Record two page views without a bearer token:

```bash
curl --fail --request POST http://127.0.0.1:8001/visitor-count
curl --fail --request POST http://127.0.0.1:8001/visitor-count
```

The second JSON `count` must be exactly one greater than the first. Requests must
match [the endpoint contract](contracts/visitor-count.md).

Verify that protected endpoints remain protected:

```bash
curl --output /dev/null --write-out '%{http_code}\n' \
  http://127.0.0.1:8001/demo-predictions
```

The response status must remain `401` without the existing bearer token.

## Validate Concurrency

Use the real-Redis integration test, which isolates and deletes its counter key
instead of polluting the displayed visitor count:

```bash
AITRAF_REDIS_URL="$AITRAF_REDIS_URL" \
  uv run --package aitraf-api pytest \
  packages/aitraf-api/tests/features/visitor_count/test_redis_integration.py
```

The test submits 100 concurrent increments and proves there are no lost or
duplicate resulting counts.

## Validate Restart Persistence

Record one returned value, then restart Redis without deleting its volume:

```bash
docker compose restart redis
docker compose ps redis
```

After Redis is healthy, record another visit. The new returned count must be one
greater than the value returned before restart.

Do not use `docker compose down --volumes`; deleting the named volume
intentionally deletes the stored counter and is outside restart persistence.

## Validate Explicit Failures

Stop Redis:

```bash
docker compose stop redis
```

A new visitor-count request must return an explicit service failure and must not
return a count. Starting the API while Redis is stopped must fail readiness.

Restart Redis before further API use:

```bash
docker compose start redis
```

The integration suite also validates malformed stored values in an isolated key
space; do not corrupt the development counter manually.

## External Deployment Handoff

Production deployment is owned by another repository and is not modified by
this feature. Its required handoff is limited to providing a reachable,
persistent Redis URL as `AITRAF_REDIS_URL` and validating the endpoint in that
environment. No production Compose, network, volume, or proxy files are added
here.
