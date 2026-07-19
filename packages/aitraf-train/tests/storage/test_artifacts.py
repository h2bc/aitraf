from pathlib import Path

import pytest

from aitraf_train.storage.artifacts import upload_directory


class StubS3Client:
    def __init__(
        self, existing_keys: set[str] | None = None, fail_on: str | None = None
    ):
        self.existing_keys = set(existing_keys or ())
        self.fail_on = fail_on
        self.uploaded: list[str] = []

    def list_objects_v2(self, *, Bucket: str, Prefix: str, MaxKeys: int):
        if Prefix in self.existing_keys:
            return {"Contents": [{"Key": Prefix}]}
        return {}

    def put_object(self, *, Bucket: str, Key: str, Body):
        if Key == self.fail_on:
            raise RuntimeError("upload rejected")
        self.uploaded.append(Key)


@pytest.fixture
def poses_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "poses"
    directory.mkdir()
    (directory / "a.npz").write_bytes(b"a")
    (directory / "b.npz").write_bytes(b"b")
    return directory


def test_uploads_every_file(poses_dir: Path) -> None:
    client = StubS3Client()

    uploaded = upload_directory(
        poses_dir, bucket="bucket", prefix="poses", force=False, s3_client=client
    )

    assert uploaded == 2
    assert client.uploaded == ["poses/a.npz", "poses/b.npz"]


def test_skips_existing_keys(poses_dir: Path) -> None:
    client = StubS3Client(existing_keys={"poses/a.npz"})

    uploaded = upload_directory(
        poses_dir, bucket="bucket", prefix="poses", force=False, s3_client=client
    )

    assert uploaded == 1
    assert client.uploaded == ["poses/b.npz"]


def test_force_overwrites_existing_keys(poses_dir: Path) -> None:
    client = StubS3Client(existing_keys={"poses/a.npz", "poses/b.npz"})

    uploaded = upload_directory(
        poses_dir, bucket="bucket", prefix="poses", force=True, s3_client=client
    )

    assert uploaded == 2
    assert client.uploaded == ["poses/a.npz", "poses/b.npz"]


def test_preserves_nested_relative_paths(tmp_path: Path) -> None:
    directory = tmp_path / "poses"
    (directory / "nested").mkdir(parents=True)
    (directory / "nested" / "c.npz").write_bytes(b"c")
    client = StubS3Client()

    upload_directory(
        directory, bucket="bucket", prefix="poses", force=False, s3_client=client
    )

    assert client.uploaded == ["poses/nested/c.npz"]


def test_raises_on_first_upload_failure(poses_dir: Path) -> None:
    client = StubS3Client(fail_on="poses/a.npz")

    with pytest.raises(RuntimeError, match="Failed to upload"):
        upload_directory(
            poses_dir, bucket="bucket", prefix="poses", force=False, s3_client=client
        )

    assert client.uploaded == []


def test_raises_when_directory_missing(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Artifact directory not found"):
        upload_directory(
            tmp_path / "absent",
            bucket="bucket",
            prefix="poses",
            force=False,
            s3_client=StubS3Client(),
        )


def test_raises_when_path_is_not_a_directory(tmp_path: Path) -> None:
    target = tmp_path / "poses.npz"
    target.write_bytes(b"a")

    with pytest.raises(RuntimeError, match="not a directory"):
        upload_directory(
            target,
            bucket="bucket",
            prefix="poses",
            force=False,
            s3_client=StubS3Client(),
        )


def test_raises_when_directory_is_empty(tmp_path: Path) -> None:
    directory = tmp_path / "poses"
    directory.mkdir()

    with pytest.raises(RuntimeError, match="No files found"):
        upload_directory(
            directory,
            bucket="bucket",
            prefix="poses",
            force=False,
            s3_client=StubS3Client(),
        )


def test_raises_when_prefix_is_empty(poses_dir: Path) -> None:
    with pytest.raises(RuntimeError, match="prefix must be provided"):
        upload_directory(
            poses_dir, bucket="bucket", prefix="", force=False, s3_client=StubS3Client()
        )
