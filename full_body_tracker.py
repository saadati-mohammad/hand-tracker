"""Full-body tracking: pose skeleton, both hands with fingers, and face mesh.

Reuses open_camera() from hand_tracker.py so the DroidCam index handling
stays in one place. Press 'q' or ESC to quit; 'f' toggles the face mesh
(478 points is the heaviest part, so it can be turned off for speed).
"""

import json
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision, BaseOptions

from hand_tracker import open_camera, CONNECTIONS as HAND_CONNECTIONS, TIP_IDS

MODEL_DIR = Path(__file__).parent
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
FACE_MODEL = MODEL_DIR / "face_landmarker.task"

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
# 0-10 head/face, 11-22 arms/torso, 23-32 legs/feet
POSE_FACE_IDS = set(range(0, 11))
POSE_TORSO_IDS = set(range(11, 23))

FACE_CONN_FILE = MODEL_DIR / "face_mesh_connections.json"
FACE_FEATURES = None  # loaded lazily


def face_features():
    """Tessellation and feature edges for the 478-point face mesh.

    mediapipe 1.x no longer ships the connection lists, so they are stored
    next to the model as a JSON asset.
    """
    global FACE_FEATURES
    if FACE_FEATURES is None:
        with open(FACE_CONN_FILE, encoding="utf-8") as fh:
            FACE_FEATURES = {k: [tuple(e) for e in v]
                             for k, v in json.load(fh).items()}
    return FACE_FEATURES


def build_landmarkers(with_face=True):
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
    )
    face = build_face_landmarker() if with_face else None
    return pose, hands, face


def build_face_landmarker():
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def to_pixels(landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]


def draw_pose(frame, landmarks, w, h):
    pts = to_pixels(landmarks, w, h)
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for i, (x, y) in enumerate(pts):
        if i in POSE_FACE_IDS:
            color, radius = (255, 255, 0), 3
        elif i in POSE_TORSO_IDS:
            color, radius = (0, 255, 0), 5
        else:
            color, radius = (255, 100, 255), 5
        cv2.circle(frame, (x, y), radius, color, -1)
    return pts


def draw_face(frame, landmarks, w, h):
    pts = to_pixels(landmarks, w, h)
    feats = face_features()
    for a, b in feats["FACEMESH_TESSELATION"]:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (70, 70, 70), 1)
    # Eyes, eyebrows, lips and jawline get their own colour so they read clearly
    # on top of the faint tessellation.
    for key, color, thick in (
        ("FACEMESH_CONTOURS", (200, 160, 0), 1),
        ("FACEMESH_LIPS", (0, 80, 255), 2),
        ("FACEMESH_LEFT_EYE", (0, 255, 0), 2),
        ("FACEMESH_RIGHT_EYE", (0, 255, 0), 2),
        ("FACEMESH_LEFT_EYEBROW", (255, 0, 255), 2),
        ("FACEMESH_RIGHT_EYEBROW", (255, 0, 255), 2),
        ("FACEMESH_FACE_OVAL", (255, 255, 255), 2),
    ):
        for a, b in feats[key]:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], color, thick)


def draw_hands(frame, result, w, h):
    for idx, landmarks in enumerate(result.hand_landmarks):
        label = "hand"
        if idx < len(result.handedness) and result.handedness[idx]:
            label = result.handedness[idx][0].category_name
        pts = to_pixels(landmarks, w, h)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for i, (x, y) in enumerate(pts):
            color = (0, 0, 255) if i in TIP_IDS else (255, 200, 0)
            cv2.circle(frame, (x, y), 7 if i in TIP_IDS else 4, color, -1)
        cv2.putText(frame, label, (pts[0][0] - 20, pts[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def angle_deg(a, b, c):
    """Angle at b formed by a-b-c, in degrees (2D)."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def joint_angles(pts):
    """Elbow, shoulder-spread and knee angles — the useful 'detail' readouts."""
    out = {}
    for name, (a, b, c) in {
        "L elbow": (11, 13, 15),
        "R elbow": (12, 14, 16),
        "L knee": (23, 25, 27),
        "R knee": (24, 26, 28),
    }.items():
        out[name] = angle_deg(pts[a], pts[b], pts[c])
    return out


def main():
    pose, hands, face = build_landmarkers(with_face=True)

    cap = open_camera()
    print("Full-body tracker running. 'q'/ESC quits, 'f' toggles the face mesh.")

    prev_t = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed; stopping.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)

            pose_result = pose.detect_for_video(mp_image, ts)
            hand_result = hands.detect_for_video(mp_image, ts)
            face_result = face.detect_for_video(mp_image, ts) if face else None

            if face_result and face_result.face_landmarks:
                draw_face(frame, face_result.face_landmarks[0], w, h)

            body_pts = None
            if pose_result.pose_landmarks:
                body_pts = draw_pose(frame, pose_result.pose_landmarks[0], w, h)

            draw_hands(frame, hand_result, w, h)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_t, 1e-6))
            prev_t = now

            lines = [
                f"fps {fps:5.1f}   hands {len(hand_result.hand_landmarks)}"
                f"   face {'on' if face else 'off'}",
            ]
            if body_pts:
                for name, val in joint_angles(body_pts).items():
                    lines.append(f"{name} {val:5.1f} deg")

            for i, text in enumerate(lines):
                y = 30 + i * 26
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 255), 2)

            cv2.imshow("Full Body Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                if face:
                    face.close()
                    face = None
                else:
                    face = build_face_landmarker()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        hands.close()
        if face:
            face.close()


if __name__ == "__main__":
    main()
