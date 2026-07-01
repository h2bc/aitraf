# Contract: API Runtime Demo Clip Download

## Purpose

Ensure the API demo clip subset is present in runtime clip storage when a
deployment explicitly enables startup hydration.

## Runtime Inputs

- API classification manifest
- API AQA manifest
- Runtime clip storage path
- Explicit demo-download enable setting
- Object storage credentials and endpoint settings when download is enabled

## Required Behavior

- When disabled, startup does not attempt object storage downloads.
- When enabled, startup selects demo clips using the same demo-video filter as
  the `/demo-videos` behavior.
- Selected manifest rows must include source clip metadata.
- Selected clips are downloaded into the same runtime clip directory used by
  inference.
- Existing clips are reused by default.
- The API is not considered ready for demo inference until all selected clips are
  present.

## Failure Behavior

- Missing manifests, empty selected demo sets, missing source metadata, invalid
  source URIs, missing credentials, failed object downloads, or unwritable
  storage fail explicitly.
- Failures identify the affected selected clip or missing input.
- Startup must not silently remove failed clips from the demo list.

## Image Build Contract

- API image build does not require local clips.
- API image build does not download clips.
- Published images include code and manifests, not generated clip artifacts.
