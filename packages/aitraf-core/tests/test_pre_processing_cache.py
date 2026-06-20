from __future__ import annotations

import torch

from aitraf_core.pre_processing.cache import (
    load_cached_payload,
    save_cached_payload,
)


def test_cached_payload_roundtrip_when_contract_matches(tmp_path) -> None:
    path = tmp_path / "feature.pt"
    contract = {"video_id": "clip.mp4", "num_clips": 2}

    save_cached_payload(
        path,
        contract=contract,
        payload={"features": torch.ones(2, 3)},
    )

    payload = load_cached_payload(path, contract)

    assert payload is not None
    assert torch.equal(payload["features"], torch.ones(2, 3))


def test_cached_payload_returns_none_when_contract_differs(tmp_path) -> None:
    path = tmp_path / "feature.pt"
    save_cached_payload(
        path,
        contract={"video_id": "clip.mp4", "num_clips": 2},
        payload={"features": torch.ones(2, 3)},
    )

    assert load_cached_payload(path, {"video_id": "clip.mp4", "num_clips": 3}) is None
