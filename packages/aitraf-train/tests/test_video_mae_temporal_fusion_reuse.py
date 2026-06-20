from __future__ import annotations

from pathlib import Path

import torch

from aitraf_train.data_ops.video_mae_feature_extraction import (
    VideoMaeFeatureExtractionConfig,
    extract_clip_features,
)


class Processor:
    pass


class Model:
    pass


def test_extract_clip_features_reuses_core_feature_extraction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    clip = tmp_path / "clips" / "video.mp4"
    clip.parent.mkdir()
    clip.touch()
    features_dir = tmp_path / "features"
    calls: list[str] = []

    def fake_cached_video_mae_feature_extraction(**kwargs):
        calls.append(kwargs["video_id"])
        feature_path = kwargs["feature_path"]
        feature_path.parent.mkdir(parents=True)
        torch.save({"features": torch.ones(2, 4)}, feature_path)
        return torch.ones(2, 4)

    monkeypatch.setattr(
        "aitraf_train.data_ops.video_mae_feature_extraction.cached_video_mae_feature_extraction",
        fake_cached_video_mae_feature_extraction,
    )

    result = extract_clip_features(
        clip=clip,
        processor=Processor(),
        model=Model(),
        config=VideoMaeFeatureExtractionConfig(
            clips_dir=clip.parent,
            features_dir=features_dir,
            backbone="backbone",
            model_cache_dir=tmp_path / "models",
            device="cpu",
            batch_size=1,
            num_workers=0,
            num_clips=2,
            sample_frames=1,
            sampling_dist="uniform",
            force=False,
            limit=None,
        ),
    )

    assert result == "processed"
    assert calls == ["video.mp4"]
    assert list(features_dir.rglob("*.pt"))
