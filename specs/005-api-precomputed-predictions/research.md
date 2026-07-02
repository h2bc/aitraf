# Research: AITRAF API Precomputed Predictions

## Decision: Download Artifacts At Startup

**Rationale**: The API should still get predictions from MLOps, but requests
should not depend on MLOps network access. Startup download is enough.
Only MLflow run IDs need runtime configuration because prediction artifact paths
are stable and should live as constants in API code.
The artifact must be a full test prediction table logged by `aitraf-train`, not
`misses_summary.json`.

**Alternatives considered**:
- Live inference: rejected because the demo only serves selected test videos.
- Commit prediction files: rejected because generated predictions belong in
  MLOps, not source control.
- Download on every request: rejected because it makes each demo request slower
  and more fragile.
- Configurable artifact path env vars: rejected because the artifact names are
  stable, and exposing them in `.env.example` adds unnecessary runtime surface.

## Decision: Match Rows At Startup And Cache The Response

**Rationale**: This is the simplest request behavior. Load the prediction
artifacts as lists, match classification and AQA rows by `video_id` once at
startup, keep the final response in memory, and return it from
`/demo-predictions`.

**Alternatives considered**:
- Pre-join artifacts into one file: unnecessary for this demo.
- Build indexes at startup: unnecessary unless the direct matching pass becomes
  a measured performance problem.

## Decision: Remove `aitraf-core` From API Runtime Only

**Rationale**: The simplified API no longer loads models or runs preprocessing,
so it does not need `aitraf-core`. `aitraf-core` should still exist in the repo
for reusable model loading and ML runtime code used outside this demo API.

**Alternatives considered**:
- Move model loading into `aitraf-train`: rejected as unrelated to API
  simplification.
- Keep `aitraf-core` in the API image: rejected because it keeps model runtime
  dependencies in a service that no longer performs inference.

## Decision: Add Tiny API Artifact Helper

**Rationale**: The API only needs to download/read prediction artifacts. Existing
core MLflow helpers load models and import Torch/Transformers, so they are the
wrong dependency for the simplified API.

**Alternatives considered**:
- Reuse `aitraf_core.loading.mlflow`: rejected because it is model-loading code.
- Put API startup download logic in `aitraf-train`: rejected because it is API
  serving behavior.

## Decision: Log Full Test Predictions From Evaluation

**Rationale**: The inspected MLflow eval runs only save metrics, params, plots,
and `misses_summary.json`. `misses_summary.json` contains only wrong
predictions, so it cannot power `/demo-predictions`. All `aitraf-train`
evaluation scripts should support logging a full test prediction artifact with
one row per test example and all display metadata needed by downstream serving.
The demo then reruns evaluation for the two registered demo models and uses the
new eval run IDs in the API.

**Alternatives considered**:
- Use `misses_summary.json`: rejected because it omits correct predictions.
- Keep reading local manifests in the API: rejected because the prediction
  artifact can carry the needed metadata and keep the API runtime simpler.
- Retrain models: rejected because this feature only needs rerunning eval for
  existing registered model artifacts unless those artifacts are unavailable or
  broken.

## Decision: Delete Live Inference Routes

**Rationale**: The API should expose the actual demo product surface: one demo
videos request with precomputed predictions.

**Alternatives considered**:
- Keep old `/inference/*` routes returning filtered rows: rejected because that
  preserves obsolete API shape.
- Deprecate old routes: rejected because this repo removes obsolete behavior
  directly instead of maintaining compatibility shims.
