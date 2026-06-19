"""Video metadata helpers."""

from __future__ import annotations

import av


def get_video_rotation_deg(path) -> int:
    with av.open(str(path)) as c:
        frame = next(c.decode(video=0))
        rot = getattr(frame, "rotation", None)
        if rot is None:
            raise ValueError("No rotation metadata found")

        return (rot + 360) % 360


__all__ = ["get_video_rotation_deg"]
