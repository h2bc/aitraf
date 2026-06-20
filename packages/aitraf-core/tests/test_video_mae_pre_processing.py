from __future__ import annotations

import torch
from torch import nn

from aitraf_core.pre_processing.models import video_mae_temporal_fusion
from aitraf_core.pre_processing.models.video_mae import (
    extract_video_mae_batch_features,
)
from aitraf_core.pre_processing.models.video_mae_temporal_fusion import (
    cached_video_mae_feature_extraction,
    video_feature_cache_dir,
    video_feature_cache_path,
)


class FeatureModel(nn.Module):
    def forward(self, *, pixel_values: torch.Tensor):
        batch = pixel_values.shape[0]
        return type(
            "Output",
            (),
            {"last_hidden_state": torch.ones(batch, 2, 3)},
        )()


def test_extract_video_mae_batch_features_shapes_temporal_clips() -> None:
    features = extract_video_mae_batch_features(
        pixel_values=torch.ones(2, 3, 1, 2, 2, 1),
        model=FeatureModel(),
        device="cpu",
        num_clips=3,
    )

    assert features.shape == (2, 3, 3)


def test_cached_video_mae_feature_extraction_loads_cached_features(
    tmp_path,
    monkeypatch,
) -> None:
    feature_cache_dir = video_feature_cache_dir(
        features_dir=tmp_path,
        backbone="test/backbone",
        num_clips=2,
        sample_frames=1,
        sampling_dist="uniform",
    )
    feature_path = video_feature_cache_path(
        feature_cache_dir=feature_cache_dir,
        video_id="clip.mp4",
    )
    feature_path.parent.mkdir(parents=True)
    torch.save({"features": torch.ones(2, 3)}, feature_path)

    def fail_extract(**kwargs):
        _ = kwargs
        raise AssertionError("cache hit should not extract")

    monkeypatch.setattr(
        video_mae_temporal_fusion,
        "extract_video_mae_clip_features",
        fail_extract,
    )

    features = cached_video_mae_feature_extraction(
        video_id="clip.mp4",
        clips_dir=tmp_path / "clips",
        feature_path=feature_path,
        feature_extractor=type(
            "Extractor",
            (),
            {"processor": object(), "model": object(), "device": "cpu"},
        )(),
        backbone="test/backbone",
        num_clips=2,
        sample_frames=1,
        sampling_dist="uniform",
        cache_video_features=True,
    )

    assert torch.equal(features, torch.ones(2, 3))


def test_cached_video_mae_feature_extraction_extracts_and_caches_missing_features(
    tmp_path,
    monkeypatch,
) -> None:
    feature_cache_dir = video_feature_cache_dir(
        features_dir=tmp_path,
        backbone="test/backbone",
        num_clips=2,
        sample_frames=1,
        sampling_dist="uniform",
    )

    def extract(**kwargs):
        _ = kwargs
        return torch.ones(2, 3), [[0], [1]]

    monkeypatch.setattr(
        video_mae_temporal_fusion,
        "extract_video_mae_clip_features",
        extract,
    )

    features = cached_video_mae_feature_extraction(
        video_id="clip.mp4",
        clips_dir=tmp_path / "clips",
        feature_path=video_feature_cache_path(
            feature_cache_dir=feature_cache_dir,
            video_id="clip.mp4",
        ),
        feature_extractor=type(
            "Extractor",
            (),
            {"processor": object(), "model": object(), "device": "cpu"},
        )(),
        backbone="test/backbone",
        num_clips=2,
        sample_frames=1,
        sampling_dist="uniform",
        cache_video_features=True,
    )

    feature_path = video_feature_cache_path(
        feature_cache_dir=feature_cache_dir,
        video_id="clip.mp4",
    )
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)

    assert torch.equal(features, torch.ones(2, 3))
    assert torch.equal(payload["features"], torch.ones(2, 3).half())


def test_cached_video_mae_feature_extraction_extracts_without_cache(
    tmp_path,
    monkeypatch,
) -> None:
    feature_cache_dir = video_feature_cache_dir(
        features_dir=tmp_path,
        backbone="test/backbone",
        num_clips=2,
        sample_frames=1,
        sampling_dist="uniform",
    )

    def extract(**kwargs):
        _ = kwargs
        return torch.ones(2, 3), [[0], [1]]

    monkeypatch.setattr(
        video_mae_temporal_fusion,
        "extract_video_mae_clip_features",
        extract,
    )

    features = cached_video_mae_feature_extraction(
        video_id="clip.mp4",
        clips_dir=tmp_path / "clips",
        feature_path=video_feature_cache_path(
            feature_cache_dir=feature_cache_dir,
            video_id="clip.mp4",
        ),
        feature_extractor=type(
            "Extractor",
            (),
            {"processor": object(), "model": object(), "device": "cpu"},
        )(),
        backbone="test/backbone",
        num_clips=2,
        sample_frames=1,
        sampling_dist="uniform",
        cache_video_features=False,
    )

    feature_path = video_feature_cache_path(
        feature_cache_dir=feature_cache_dir,
        video_id="clip.mp4",
    )

    assert torch.equal(features, torch.ones(2, 3))
    assert not feature_path.exists()
