# aitraf-api

FastAPI package for serving the AITRAF demo API.

The API uses precomputed prediction artifacts from MLflow. It does not load
models, decode videos, run feature extraction, or perform live inference at
request time.

At startup, the app loads configured prediction artifacts and prepares demo
prediction records in memory. When serving those records, it uses API-owned S3
configuration to generate browser-playable presigned video URLs.

## Configuration

Runtime configuration is environment-based. See the root `.env.example` for the
current variable names.

Required configuration categories:

- API authentication token
- MLflow tracking URI and credentials
- classification and AQA prediction run IDs
- S3-compatible endpoint, bucket, region, and credentials

S3 credentials stay in the API runtime. Clients receive presigned HTTP(S) video
URLs rather than raw S3 paths or credentials.

## Package Layout

- `src/aitraf_api/app.py`: FastAPI app factory and startup wiring
- `src/aitraf_api/config.py`: environment-backed settings
- `src/aitraf_api/auth.py`: bearer-token authentication
- `src/aitraf_api/schemas.py`: public response schemas
- `src/aitraf_api/features/health/`: health route
- `src/aitraf_api/features/demo_predictions/`: prediction artifact loading,
  response preparation, and route registration

## Development

Run the API through the repository task:

```bash
task api:run
```

Build the API image from the repository root:

```bash
docker build -f packages/aitraf-api/Dockerfile -t aitraf-api:local .
```

The image is CPU-only and does not include live inference dependencies.
