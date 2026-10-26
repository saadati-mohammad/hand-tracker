"""Camera discovery and capture."""

from __future__ import annotations

import os
import sys

import cv2

DEFAULT_BACKEND = cv2.CAP_DSHOW
# DroidCam and other virtual cameras do not reliably land on index 0, and on
# Windows a device can report isOpened() while never delivering a frame, so
# candidates are probed with a real read() before being accepted.
CANDIDATE_INDICES = (0, 1, 2, 3, 4)


class CameraError(RuntimeError):
    pass


def _probe(index: int, backend: int) -> cv2.VideoCapture | None:
    cap = cv2.VideoCapture(index, backend)
    if cap.isOpened():
        ok, _ = cap.read()
        if ok:
            return cap
    cap.release()
    return None


def open_camera(index: int | None = None, backend: int = DEFAULT_BACKEND):
    """Open a camera that actually delivers frames.

    ``index`` may be given explicitly; otherwise the HAND_TRACKER_CAMERA env
    var, a command-line argument, then the candidate list are tried in turn.
    """
    if index is None:
        env = os.environ.get("HAND_TRACKER_CAMERA")
        if env:
            index = int(env)
    if index is None and len(sys.argv) > 1 and sys.argv[1].isdigit():
        index = int(sys.argv[1])

    candidates = (index,) if index is not None else CANDIDATE_INDICES
    for candidate in candidates:
        cap = _probe(candidate, backend)
        if cap is not None:
            return cap
    raise CameraError(
        f"No working camera found (tried {list(candidates)}). "
        "Pass an index, e.g. --camera 2, or set HAND_TRACKER_CAMERA."
    )


def list_cameras(max_index: int = 6) -> list[int]:
    """Indices that currently produce frames. Used by the camera picker."""
    found = []
    for index in range(max_index):
        cap = _probe(index, DEFAULT_BACKEND)
        if cap is not None:
            found.append(index)
            cap.release()
    return found


def configure(cap, width: int | None = None, height: int | None = None):
    """Request a resolution; the driver may ignore or clamp it."""
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


def actual_resolution(cap) -> tuple[int, int]:
    return (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )