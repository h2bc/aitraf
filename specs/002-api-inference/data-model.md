# Data Model: API Inference Surface

## DemoVideo

Represents one selectable video shown in the demo UI.

**Fields**:

- `id`: Stable id exposed by the API for inference selection. For this feature, this is the manifest `video_id`.
- `video_id`: Video filename from the manifest.
- `s3_path`: Optional source path from the manifest.
- `person`: Optional skater/person metadata from the manifest.
- `trick`: Optional trick label from the manifest, display-safe for demo selection.
- `execution_score`: Optional quality score label from the manifest, display-safe for demo selection.

**Validation Rules**:

- `id` and `video_id` are required.
- The list is built from matching videos in the current classification and AQA test manifests.
- The response is for display and selection only; it does not expose task grouping.
- The API must not fabricate videos if a manifest is missing or unreadable.

## DemoVideosResponse

Container returned by the demo-video listing endpoint.

**Fields**:

- `videos`: Flat list of `DemoVideo` values used by the frontend to display selectable demo videos.

**Validation Rules**:

- The response is not split by task.
- Each returned video is selectable by id for either inference endpoint when that endpoint can resolve the id in its current test manifest.
- Missing required manifests are explicit errors, not partial successful responses.

## InferenceRequestSelection

Represents the only client-controlled input for an inference request.

**Fields**:

- `id`: Stable video id returned by the demo-video listing endpoint.

**Validation Rules**:

- `id` is required.
- The endpoint being called determines which model and manifest are used.
- The id must resolve to a current manifest row by `video_id` for that endpoint.
- The id does not select arbitrary uploaded files or caller-provided paths.

## InferenceResult

Represents the demo-facing prediction result for one selected video.

**Fields**:

- `video_id`: Selected video id.
- `prediction`: User-formatted prediction result.
- `ground_truth`: User-formatted ground truth result from the current manifest.
- `model`: Minimal model metadata needed for the demo UI, such as model family or variant.

**Prediction Fields**:

- Trick classification prediction includes the predicted trick label and model confidence, not an internal class id.
- Trick AQA prediction includes the predicted assessment/score label or value and model confidence, not an internal class id.
- Prediction confidence is required in inference responses.

**Ground Truth Fields**:

- Trick classification ground truth includes the manifest trick label.
- Trick AQA ground truth includes the manifest assessment/score label or value.

**Model Fields**:

- `kind`: Short user-facing model kind, such as `ordinal`, `video_mae`, or `fusion`.

**Validation Rules**:

- The response must correspond to the selected video id and endpoint.
- Prediction and ground truth values must be decoded to user-facing labels/values before returning.
- Prediction confidence must be included as a normalized value from 0 to 1.
- Missing model metadata, image processor, video file, or artifact is an explicit service error.
- The API tests do not validate prediction quality, only response shape and orchestration.

## RegisteredModelReference

Defines the minimal API-owned reference to the registered model used by one
inference endpoint.

**Defined by `.env` / environment**:

- `MLFLOW_TRACKING_URI`.
- API auth token.
- Any MLflow credentials required by the deployment.
- `classification_model_uri`: Required registered MLflow model URI for trick classification, for example `models:/aitraf-trick-classification@infant`.
- `aqa_model_uri`: Required registered MLflow model URI for trick AQA, for example `models:/aitraf-trick-aqa@infant`.
- `data_path`: Required repo data directory.
- `storage_path`: Required repo storage directory.

**Defined by API config/constants**:

- Endpoint-to-model URI mapping.
- Compact demo-facing model kind, such as `ordinal`, `video_mae`, or `fusion`.
- `classification_manifest_path`: Derived from `data_path`.
- `aqa_manifest_path`: Derived from `data_path`.
- `clips_dir`: Derived from `storage_path`.

**Loaded from the registered MLflow model package**:

- Trained model artifact.
- Processor.
- Preprocessing metadata available to the shared core inference path, including frame count and sampling strategy.
- Output decoding metadata, including label/value mapping needed for user-facing prediction results.

**Fields**:

- `endpoint`: Inference endpoint this reference belongs to.
- `model_uri`: Registered MLflow model URI to load.
- `model_kind`: Compact demo-facing model kind to return.
- `manifest_path`: Current test manifest used to resolve the requested `video_id`.

**Validation Rules**:

- The API must not define processor, frame count, sampling strategy, or label decoding as API constants.
- The registered model/core inference path must provide processor, preprocessing metadata, and output decoding metadata.
- If a registered model is missing required inference metadata, implementation must fix model registration/logging instead of duplicating those values in API code.
- Missing data path, storage path, model reference, MLflow environment, manifest, clips path, registered model, or required registered-model metadata must fail explicitly.

## AuthToken

Represents the configured app token used for protected endpoints.

**Fields**:

- `token`: Secret value supplied through environment/configuration.

**Validation Rules**:

- Token is never committed to repository files.
- Protected endpoints reject missing or invalid tokens.
- Health may remain unauthenticated for service monitoring.
