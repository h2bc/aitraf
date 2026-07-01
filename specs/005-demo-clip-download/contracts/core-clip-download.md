# Contract: Shared Core Clip Download

## Purpose

Provide a reusable clip download surface for both API runtime hydration and
train data preparation without either package duplicating S3 mechanics.

## Inputs

- One or more clip download requests
- Destination clip directory
- Optional force/refresh behavior
- Object storage client or explicit object storage settings

## Required Behavior

- Parse supported object storage URIs and reject malformed or unsupported
  sources.
- Create destination directories when permitted by the filesystem.
- Skip existing destination files unless force/refresh behavior is requested.
- Download missing or forced clips to the requested destination path.
- Return successful outcomes for downloaded and skipped-existing clips.
- Raise explicit errors for missing credentials, inaccessible objects, failed
  downloads, or unwritable destinations.

## Prohibited Behavior

- Must not import API or train packages.
- Must not know about API demo selection.
- Must not know about Hydra configs, label schemas, or train pipeline progress
  reporting.
- Must not silently skip failed downloads.

## Validation Expectations

- Unit tests cover URI parsing, destination path construction, skip-existing
  behavior, force behavior, and error propagation.
- Tests use a fake or mocked object storage client where practical.
