# Data Model: Demo Clip Download

## S3 Settings

Represents the object storage connection settings used by shared clip download
behavior.

**Fields**:

- `endpoint_url`: S3-compatible endpoint URL
- `region_name`: S3-compatible region
- `access_key`: object storage access key
- `secret_key`: object storage secret key
- `bucket`: optional default bucket for operations that require a configured
  bucket

**Validation rules**:

- Endpoint, region, access key, and secret key are required for client-backed
  downloads.
- Bucket is required only for operations that are bucket-relative rather than
  full-URI based.
- Missing required settings raise explicit errors.

## Clip Download Request

Represents one clip that should be present in local clip storage.

**Fields**:

- `video_id`: destination filename or relative clip identifier
- `source_uri`: full source object URI for the clip
- `destination_dir`: clip storage directory supplied by the caller
- `force`: whether to overwrite an existing local clip

**Relationships**:

- Built from train label rows for offline data preparation.
- Built from API demo manifest rows for runtime demo hydration.
- Executed by the shared core downloader.

**Validation rules**:

- `video_id` must be present and must identify the destination clip name.
- `source_uri` must be present and parse as a supported object storage URI.
- `destination_dir` must be writable or creatable by the runtime.
- Existing files are reused when `force` is false.

## Clip Download Result

Represents the outcome of attempting to satisfy a clip download request.

**Fields**:

- `video_id`: requested clip identifier
- `destination_path`: local clip path
- `status`: downloaded or skipped-existing

**Validation rules**:

- Failed downloads do not produce successful results.
- Download failures identify the affected source and destination.

## Demo Clip Hydration Request

Represents API startup work needed to ensure selected demo clips exist.

**Fields**:

- `enabled`: explicit runtime setting controlling whether hydration runs
- `classification_manifest_path`: manifest used by demo classification selection
- `aqa_manifest_path`: manifest used by demo AQA selection
- `clips_dir`: runtime clip storage directory
- `force`: optional refresh behavior

**Relationships**:

- Reads manifest rows through existing API manifest validation.
- Uses API demo filtering to select video IDs.
- Converts selected manifest rows into clip download requests.

**State transitions**:

1. Disabled: no download work is attempted.
2. Enabled pending: manifests and selected rows are read.
3. Downloading: shared core downloader satisfies selected clip requests.
4. Ready: all selected clips exist locally.
5. Failed: any selected clip cannot be resolved, downloaded, or written.

## Train Clip Download Adaptation

Represents the existing train data-op's conversion from labels to shared clip
download requests.

**Fields**:

- `labels_path`: input labels JSONL path
- `output_dir`: train clip storage directory
- `force`: existing train force behavior
- `clip_source_column`: label field containing source clip URI

**Validation rules**:

- Labels path must exist and be readable.
- Source column must exist and contain supported object storage URIs.
- Existing train output paths remain compatible with current data pipeline
  expectations.
