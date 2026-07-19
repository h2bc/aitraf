"""Pose overlay rendering shared by extraction, notebooks, and the API."""

from aitraf_core.render.draw import (
    POSE_DEFAULT_SKELETON,
    draw_bounding_boxes,
    draw_pose_keypoints,
)
from aitraf_core.render.poses import (
    PoseArtifact,
    PoseArtifactError,
    load_pose_artifact,
)
from aitraf_core.render.video import (
    RenderError,
    draw_pose_frame,
    get_frame_rate,
    get_video_rotation_deg,
    iter_frames,
    render_pose_video,
)

__all__ = [
    "POSE_DEFAULT_SKELETON",
    "PoseArtifact",
    "PoseArtifactError",
    "RenderError",
    "draw_bounding_boxes",
    "draw_pose_frame",
    "draw_pose_keypoints",
    "get_frame_rate",
    "get_video_rotation_deg",
    "iter_frames",
    "load_pose_artifact",
    "render_pose_video",
]
