"""MediaPipe landmarker lifecycle and per-frame inference."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .paths import asset

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
HAND_TIPS = (4, 8, 12, 16, 20)

_face_features: dict[str, list[tuple[int, int]]] | None = None


def face_features() -> dict[str, list[tuple[int, int]]]:
    """Tessellation and feature edges for the 478-point face mesh.

    mediapipe 1.x no longer ships mediapipe.python.solutions, so the
    connection lists are stored beside the models as JSON.
    """
    global _face_features
    if _face_features is None:
        with open(asset("face_mesh_connections.json"), encoding="utf-8") as fh:
            raw = json.load(fh)
        _face_features = {k: [tuple(e) for e in v] for k, v in raw.items()}
    return _face_features


@dataclass
class Frame:
    """Everything the pipeline produced for one frame, in normalized coords."""

    image: np.ndarray
    pose: list | None = None
    hands: list[list] = field(default_factory=list)
    handedness: list[str] = field(default_factory=list)
    face: list | None = None

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]


def to_pixels(landmarks, width: int, height: int) -> list[tuple[int, int]]:
    return [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]


class Tracker:
    """Owns the pose, hand and face landmarkers and runs them on frames."""

    def __init__(self, use_face: bool = True, num_hands: int = 2):
        self.use_face = use_face
        self.num_hands = num_hands
        self._pose = self._build_pose()
        self._hands = self._build_hands()
        self._face = self._build_face() if use_face else None
        self._last_ts = -1

    # -- construction ----------------------------------------------------
    @staticmethod
    def _build_pose():
        return vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(asset("pose_landmarker_lite.task"))
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    def _build_hands(self):
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(asset("hand_landmarker.task"))
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=self.num_hands,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
        )

    def _build_face(self):
        return vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(asset("face_landmarker.task"))
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    # -- runtime ---------------------------------------------------------
    def set_face_enabled(self, enabled: bool):
        """Create or release the face landmarker on demand (it is the costliest)."""
        if enabled and self._face is None:
            self._face = self._build_face()
        elif not enabled and self._face is not None:
            self._face.close()
            self._face = None
        self.use_face = enabled

    def _timestamp(self, ts_ms: int) -> int:
        """Video mode requires strictly increasing timestamps."""
        if ts_ms <= self._last_ts:
            ts_ms = self._last_ts + 1
        self._last_ts = ts_ms
        return ts_ms

    def process(self, frame_bgr: np.ndarray, ts_ms: int) -> Frame:
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        )
        ts = self._timestamp(ts_ms)

        result = Frame(image=frame_bgr)

        pose_result = self._pose.detect_for_video(mp_image, ts)
        if pose_result.pose_landmarks:
            result.pose = pose_result.pose_landmarks[0]

        hand_result = self._hands.detect_for_video(mp_image, ts)
        result.hands = list(hand_result.hand_landmarks)
        result.handedness = [
            cats[0].category_name if cats else "hand"
            for cats in hand_result.handedness
        ]

        if self._face is not None:
            face_result = self._face.detect_for_video(mp_image, ts)
            if face_result.face_landmarks:
                result.face = face_result.face_landmarks[0]

        return result

    def close(self):
        for landmarker in (self._pose, self._hands, self._face):
            if landmarker is not None:
                landmarker.close()
        self._face = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()