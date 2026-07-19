"""Drawing helpers used in preprocessing and visualization.

Pose styling matches Ultralytics' own pose plotting so overlays look the same
whether they come from `results.plot()` in a notebook or from a rendered demo
clip. The skeleton layout and colour tables below are the values Ultralytics
uses; they are plain data, so reproducing them here avoids pulling the training
stack into the serving image.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# COCO-17 skeleton, 0-indexed pairs.
POSE_DEFAULT_SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
)

_GREEN = (0, 255, 0)
_ORANGE = (255, 128, 0)
_BLUE = (51, 153, 255)
_MAGENTA = (255, 51, 255)

# Per-limb colours, aligned to POSE_DEFAULT_SKELETON.
LIMB_COLORS: tuple[tuple[int, int, int], ...] = (
    (_BLUE,) * 4 + (_MAGENTA,) * 3 + (_ORANGE,) * 5 + (_GREEN,) * 7
)

# Per-keypoint colours: head green, arms orange, legs blue.
KEYPOINT_COLORS: tuple[tuple[int, int, int], ...] = (
    (_GREEN,) * 5 + (_ORANGE,) * 6 + (_BLUE,) * 6
)

# Keypoints below this confidence are predictions the model is not sure about
# (commonly facial landmarks on a rider filmed from behind). Ultralytics hides
# them; drawing them scatters stray points around the head.
DEFAULT_CONF_THRESHOLD = 0.25


def line_width_for(width: int, height: int) -> int:
    """Scale stroke width to the frame, as Ultralytics does."""
    return max(round((width + height) / 2 * 0.003), 2)


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
    scores: Iterable[Sequence[float]],
    *,
    width: int,
    height: int,
    skeleton: Iterable[tuple[int, int]] | None = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    line_width: int | None = None,
    point_radius: int | None = None,
) -> None:
    """Draw pose keypoints and skeleton edges on a PIL canvas.

    `scores` carries one confidence per keypoint; points below
    `conf_threshold`, and edges with either endpoint below it, are omitted.
    """
    stroke = line_width if line_width is not None else line_width_for(width, height)
    radius = point_radius if point_radius is not None else stroke
    edge_width = max(1, math.ceil(stroke / 2))
    edges = tuple(skeleton) if skeleton is not None else POSE_DEFAULT_SKELETON

    for pose, pose_scores in zip(poses, scores):
        points = [
            (kp[0] * width, kp[1] * height, float(conf))
            for kp, conf in zip(pose, pose_scores)
        ]

        for index, (x, y, conf) in enumerate(points):
            if conf < conf_threshold or not _is_detected(x, y):
                continue
            color = KEYPOINT_COLORS[index % len(KEYPOINT_COLORS)]
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=color,
            )

        for edge_index, (start, end) in enumerate(edges):
            x1, y1, conf1 = points[start]
            x2, y2, conf2 = points[end]
            if conf1 < conf_threshold or conf2 < conf_threshold:
                continue
            if not _is_detected(x1, y1) or not _is_detected(x2, y2):
                continue
            draw.line(
                [x1, y1, x2, y2],
                fill=LIMB_COLORS[edge_index % len(LIMB_COLORS)],
                width=edge_width,
            )


def _is_detected(x: float, y: float) -> bool:
    """Zero coordinates mark a keypoint the model did not place."""
    return x > 0 and y > 0


__all__ = [
    "DEFAULT_CONF_THRESHOLD",
    "KEYPOINT_COLORS",
    "LIMB_COLORS",
    "POSE_DEFAULT_SKELETON",
    "draw_bounding_boxes",
    "draw_pose_keypoints",
    "line_width_for",
]
