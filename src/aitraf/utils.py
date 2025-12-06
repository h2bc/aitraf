"""Utility helpers used across visualization and preprocessing."""

from __future__ import annotations

from typing import Iterable, Sequence

import av


POSE_DEFAULT_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 12),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def get_video_rotation_deg(path) -> int:
    with av.open(str(path)) as c:
        frame = next(c.decode(video=0))
        rot = getattr(frame, "rotation", None)
        if rot is None:
            raise ValueError("No rotation metadata found")

        return (rot + 360) % 360


def draw_bounding_boxes(
    draw,
    boxes: Iterable[Sequence[float]],
    *,
    width: int,
    height: int,
    color: str = "red",
    line_width: int = 2,
) -> None:
    """Draw YOLO-format bounding boxes on a PIL ImageDraw canvas."""

    for box in boxes:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = box[:4]
        draw.rectangle(
            [x1 * width, y1 * height, x2 * width, y2 * height],
            outline=color,
            width=line_width,
        )


def draw_pose_keypoints(
    draw,
    poses: Iterable[Sequence[Sequence[float]]],
    *,
    width: int,
    height: int,
    skeleton: Iterable[tuple[int, int]] | None = None,
    color: str = "lime",
    line_width: int = 2,
    point_radius: int = 3,
) -> None:
    """Draw pose keypoints and optional skeleton edges on a PIL canvas."""

    for pose in poses:
        for kp in pose:
            x, y = kp[:2]
            cx, cy = x * width, y * height
            draw.ellipse(
                [
                    cx - point_radius,
                    cy - point_radius,
                    cx + point_radius,
                    cy + point_radius,
                ],
                outline=color,
                width=line_width,
            )

        if skeleton is None:
            continue

        for a, b in skeleton:
            x1p, y1p = pose[a][0] * width, pose[a][1] * height
            x2p, y2p = pose[b][0] * width, pose[b][1] * height
            draw.line([x1p, y1p, x2p, y2p], fill=color, width=line_width)
