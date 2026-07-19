"""Render tests.

The rotation cases are the important ones: pose keypoints are normalized against
the rotated frame, so a renderer that skips or mis-orders the rotation produces a
plausible-looking video with the skeleton in the wrong place. That failure is
invisible to every check that does not assert pixel positions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from aitraf_core.render.poses import PoseArtifact
from aitraf_core.render.video import (
    VIDEO_PLUGIN,
    RenderError,
    get_frame_rate,
    get_video_rotation_deg,
    render_pose_video,
)

WIDTH, HEIGHT, FRAMES, FPS = 64, 32, 6, 10


def _make_clip(path: Path, *, rotation: int | None = None) -> Path:
    plain = path.parent / "plain.mp4"
    # Real clips always carry audio and the renderer muxes it back in, so the
    # fixture needs an audio track for the render to be representative.
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:size={WIDTH}x{HEIGHT}:rate={FPS}:duration={FRAMES / FPS}",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=44100:cl=stereo:d={FRAMES / FPS}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            "-y",
            str(plain),
        ],
        check=True,
        capture_output=True,
    )
    if rotation is None:
        return plain

    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-display_rotation",
            str(rotation),
            "-i",
            str(plain),
            "-c",
            "copy",
            "-y",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _artifact(
    *, n_frames: int = FRAMES, point: tuple[float, float] | None = None
) -> PoseArtifact:
    """Build an artifact placing all 17 keypoints at one normalized point."""
    if point is None:
        keypoints = [np.zeros((0, 17, 2), dtype=np.float32) for _ in range(n_frames)]
        scores = [np.zeros((0, 17), dtype=np.float32) for _ in range(n_frames)]
    else:
        keypoints = [
            np.full((1, 17, 2), point, dtype=np.float32) for _ in range(n_frames)
        ]
        scores = [np.ones((1, 17), dtype=np.float32) for _ in range(n_frames)]

    return PoseArtifact(
        frames=np.arange(n_frames, dtype=np.int32),
        keypoints=np.array(keypoints, dtype=object),
        scores=np.array(scores, dtype=object),
    )


def _overlay_centroid(frame: np.ndarray) -> tuple[float, float]:
    """Return the (x, y) centroid of drawn overlay pixels.

    The fixture clip is solid black, so any bright pixel is overlay. Keypoints
    and limbs are multicoloured, so this must not key on one hue.
    """
    mask = frame.max(axis=2) > 80
    ys, xs = np.nonzero(mask)
    assert xs.size, "no overlay pixels found"
    return float(xs.mean()), float(ys.mean())


def test_get_video_rotation_deg_normalizes_negative(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "rot.mp4", rotation=-90)
    assert get_video_rotation_deg(clip) == 270


def test_get_video_rotation_deg_is_zero_without_metadata(tmp_path: Path) -> None:
    """PyAV reports 0 for an unrotated clip; this matches the extraction helper."""
    clip = _make_clip(tmp_path / "plain.mp4")
    assert get_video_rotation_deg(clip) == 0


def test_get_frame_rate(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "rot.mp4", rotation=90)
    assert get_frame_rate(clip) == pytest.approx(FPS)


def test_render_preserves_frame_count_and_orientation(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "rot.mp4", rotation=90)
    destination = tmp_path / "out.mp4"

    render_pose_video(clip, _artifact(point=(0.5, 0.5)), destination)

    frames = list(iio.imiter(destination, plugin=VIDEO_PLUGIN))
    assert len(frames) == FRAMES
    # 90 degrees of rotation swaps the stored dimensions.
    assert frames[0].shape[:2] == (WIDTH, HEIGHT)


def test_render_places_overlay_in_rotated_coordinate_space(tmp_path: Path) -> None:
    """A renderer that ignores rotation puts the skeleton somewhere else."""
    clip = _make_clip(tmp_path / "rot.mp4", rotation=90)
    destination = tmp_path / "out.mp4"
    # Deliberately off-centre so a wrong orientation cannot coincidentally match.
    point = (0.25, 0.75)

    render_pose_video(clip, _artifact(point=point), destination)

    frame = next(iter(iio.imiter(destination, plugin=VIDEO_PLUGIN)))
    x, y = _overlay_centroid(frame)

    # Expected dimensions are stated explicitly rather than read back from the
    # frame: deriving them from the output would make this assertion hold for
    # any orientation, which is exactly the bug it exists to catch.
    rotated_width, rotated_height = HEIGHT, WIDTH
    assert frame.shape[:2] == (rotated_height, rotated_width)
    assert x == pytest.approx(point[0] * rotated_width, abs=3)
    assert y == pytest.approx(point[1] * rotated_height, abs=3)


def test_render_writes_frames_without_detections(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "rot.mp4", rotation=90)
    destination = tmp_path / "out.mp4"

    render_pose_video(clip, _artifact(point=None), destination)

    frames = list(iio.imiter(destination, plugin=VIDEO_PLUGIN))
    assert len(frames) == FRAMES
    assert not (frames[0].max(axis=2) > 80).any()


def test_render_rejects_artifact_shorter_than_clip(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "rot.mp4", rotation=90)

    with pytest.raises(RenderError, match="more frames than its pose artifact"):
        render_pose_video(
            clip, _artifact(n_frames=FRAMES - 2, point=(0.5, 0.5)), tmp_path / "out.mp4"
        )


def test_render_rejects_artifact_longer_than_clip(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "rot.mp4", rotation=90)

    with pytest.raises(RenderError, match="but its pose artifact has"):
        render_pose_video(
            clip, _artifact(n_frames=FRAMES + 2, point=(0.5, 0.5)), tmp_path / "out.mp4"
        )


def test_render_leaves_unrotated_clip_in_stored_orientation(tmp_path: Path) -> None:
    clip = _make_clip(tmp_path / "plain.mp4")
    destination = tmp_path / "out.mp4"

    render_pose_video(clip, _artifact(point=(0.5, 0.5)), destination)

    frame = next(iter(iio.imiter(destination, plugin=VIDEO_PLUGIN)))
    assert frame.shape[:2] == (HEIGHT, WIDTH)


def test_render_omits_low_confidence_keypoints(tmp_path: Path) -> None:
    """Keypoints the model is unsure about must not be drawn.

    Roughly 44% of face keypoints in the real corpus score below the threshold;
    drawing them scatters stray points around the rider's head.
    """
    clip = _make_clip(tmp_path / "rot.mp4", rotation=90)
    destination = tmp_path / "out.mp4"

    artifact = _artifact(point=(0.5, 0.5))
    artifact = PoseArtifact(
        frames=artifact.frames,
        keypoints=artifact.keypoints,
        scores=np.array(
            [np.zeros((1, 17), dtype=np.float32) for _ in range(FRAMES)],
            dtype=object,
        ),
    )

    render_pose_video(clip, artifact, destination)

    frame = next(iter(iio.imiter(destination, plugin=VIDEO_PLUGIN)))
    assert not (frame.max(axis=2) > 80).any()


def _stream_types(path: Path) -> set[str]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def test_render_keeps_the_original_audio(tmp_path: Path) -> None:
    """The overlay pass is video-only, so audio must be muxed back in."""
    clip = _make_clip(tmp_path / "rot.mp4", rotation=90)
    destination = tmp_path / "out.mp4"

    render_pose_video(clip, _artifact(point=(0.5, 0.5)), destination)

    assert _stream_types(destination) == {"video", "audio"}


def test_render_rejects_clip_without_audio(tmp_path: Path) -> None:
    """A silent published clip would be a silent regression; fail instead."""
    mute = tmp_path / "mute.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:size={WIDTH}x{HEIGHT}:rate={FPS}:duration={FRAMES / FPS}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(mute),
        ],
        check=True,
        capture_output=True,
    )

    with pytest.raises(RenderError, match="No audio stream"):
        render_pose_video(mute, _artifact(point=(0.5, 0.5)), tmp_path / "out.mp4")
