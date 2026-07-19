"""Decode clips, draw pose overlays, and write overlay videos.

Shared by the pose extraction pipeline, the analysis notebooks, and the demo
serving API so all three see identical frames. `iter_frames` is the single
definition of "the i-th frame of a clip": pose keypoints are normalized against
what it yields, so any surface that decodes differently silently misplaces every
overlay.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import av
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from aitraf_core.render.draw import POSE_DEFAULT_SKELETON, draw_pose_keypoints
from aitraf_core.render.poses import PoseArtifact

# PyAV is pinned rather than left to imageio's plugin auto-selection: the FFMPEG
# plugin reports a different frame count for these clips and auto-applies
# container rotation, which breaks frame alignment and double-rotates the frame.
VIDEO_PLUGIN = "pyav"
VIDEO_CODEC = "libx264"


class RenderError(RuntimeError):
    """Raised when a clip cannot be decoded or an overlay cannot be written."""


def get_video_rotation_deg(path: Path) -> int:
    """Return the container rotation in degrees, raising when it is absent.

    Keypoints are normalized against the rotated frame, so a missing value
    cannot be defaulted to zero without misplacing every overlay.
    """
    try:
        with av.open(str(path)) as container:
            frame = next(container.decode(video=0))
            rotation_deg = getattr(frame, "rotation", None)
    except Exception as exc:
        raise RenderError(f"Failed to read rotation metadata: {path}") from exc

    if rotation_deg is None:
        raise RenderError(f"No rotation metadata found: {path}")
    return (int(rotation_deg) + 360) % 360


def get_frame_rate(path: Path) -> float:
    """Return the average frame rate of the clip's video stream."""
    try:
        with av.open(str(path)) as container:
            rate = container.streams.video[0].average_rate
    except Exception as exc:
        raise RenderError(f"Failed to read frame rate: {path}") from exc

    if not rate:
        raise RenderError(f"No frame rate reported: {path}")
    return float(rate)


def iter_frames(clip_path: Path, rotation_deg: int) -> Iterator[np.ndarray]:
    """Yield display-oriented frames for a clip.

    This is the reference decode for the repository. Pose keypoints are
    normalized against these frames.
    """
    rotate_quarter_turns = rotation_deg // 90

    for frame in iio.imiter(str(clip_path), plugin=VIDEO_PLUGIN):
        if rotate_quarter_turns:
            yield np.rot90(frame, k=rotate_quarter_turns)
        else:
            yield frame


def draw_pose_frame(frame: np.ndarray, frame_keypoints, frame_scores) -> np.ndarray:
    """Return `frame` with one frame's pose detections drawn on it."""
    image = Image.fromarray(np.ascontiguousarray(frame))
    poses = np.asarray(frame_keypoints)

    if poses.size:
        draw_pose_keypoints(
            ImageDraw.Draw(image),
            poses,
            np.asarray(frame_scores),
            width=image.width,
            height=image.height,
            skeleton=POSE_DEFAULT_SKELETON,
        )

    return np.asarray(image)


def render_pose_video(source: Path, artifact: PoseArtifact, destination: Path) -> None:
    """Draw `artifact` onto `source` and write an H.264 MP4 to `destination`.

    The overlay pass produces video only, so the source's audio is muxed back in
    afterwards; the published clip keeps its original sound.
    """
    with tempfile.TemporaryDirectory(prefix="aitraf-render-") as temp_dir:
        silent = Path(temp_dir) / "video.mp4"
        _render_frames(source, artifact, silent)
        _mux_original_audio(silent, source, destination)


def _render_frames(source: Path, artifact: PoseArtifact, destination: Path) -> None:
    """Write the overlay video stream, without audio."""
    rotation_deg = get_video_rotation_deg(source)
    fps = get_frame_rate(source)

    rendered = 0
    try:
        with iio.imopen(destination, "w", plugin=VIDEO_PLUGIN) as writer:
            writer.init_video_stream(VIDEO_CODEC, fps=fps)

            for index, frame in enumerate(iter_frames(source, rotation_deg)):
                if index >= len(artifact):
                    raise RenderError(
                        f"Clip {source} has more frames than its pose artifact "
                        f"({len(artifact)})"
                    )

                writer.write_frame(
                    draw_pose_frame(
                        frame,
                        artifact.keypoints[index],
                        artifact.scores[index],
                    )
                )
                rendered += 1
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError(f"Failed to render pose overlay for {source}") from exc

    if rendered != len(artifact):
        raise RenderError(
            f"Clip {source} decoded {rendered} frames but its pose artifact has "
            f"{len(artifact)}"
        )


def _mux_original_audio(video: Path, source: Path, destination: Path) -> None:
    """Combine the rendered video stream with the source's audio stream.

    Both streams are copied, so neither the overlay nor the audio is re-encoded.
    """
    if not _has_audio_stream(source):
        raise RenderError(f"No audio stream to carry over: {source}")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video),
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c",
                "copy",
                "-shortest",
                "-y",
                str(destination),
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RenderError(f"Failed to mux audio into {destination}") from exc

    if not destination.is_file() or destination.stat().st_size == 0:
        raise RenderError(f"FFmpeg did not write {destination}")


def _has_audio_stream(path: Path) -> bool:
    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except Exception as exc:
        raise RenderError(f"Failed to inspect audio streams: {path}") from exc


__all__ = [
    "RenderError",
    "VIDEO_CODEC",
    "VIDEO_PLUGIN",
    "draw_pose_frame",
    "get_frame_rate",
    "get_video_rotation_deg",
    "iter_frames",
    "render_pose_video",
]
