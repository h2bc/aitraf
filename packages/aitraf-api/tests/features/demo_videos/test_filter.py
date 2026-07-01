from __future__ import annotations

from aitraf_api.features.demo_videos.filter import filter_demo_video_ids


def test_filter_demo_video_ids_uses_manifest_intersection_in_classification_order():
    classification_rows = [
        {"video_id": "clip-a.mp4"},
        {"video_id": "clip-b.mp4"},
        {"video_id": "clip-a.mp4"},
        {"video_id": "clip-c.mp4"},
    ]
    aqa_rows = [
        {"video_id": "clip-c.mp4"},
        {"video_id": "clip-a.mp4"},
    ]

    assert filter_demo_video_ids(
        classification_rows=classification_rows,
        aqa_rows=aqa_rows,
    ) == ["clip-a.mp4", "clip-c.mp4"]
