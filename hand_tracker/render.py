"""All drawing: skeleton, hand landmarks, face mesh, and the HUD."""

from __future__ import annotations

import cv2

from . import geometry as g
from .models import (
    HAND_CONNECTIONS,
    HAND_TIPS,
    POSE_CONNECTIONS,
    face_features,
    to_pixels,
)

POSE_HEAD = set(range(0, 11))
POSE_TORSO = set(range(11, 23))


def draw_pose(frame, pose):
    h, w = frame.shape[:2]
    pts = to_pixels(pose, w, h)
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for i, (x, y) in enumerate(pts):
        if i in POSE_HEAD:
            color, radius = (255, 255, 0), 3
        elif i in POSE_TORSO:
            color, radius = (0, 255, 0), 5
        else:
            color, radius = (255, 100, 255), 5
        cv2.circle(frame, (x, y), radius, color, -1)
    return pts


def draw_hands(frame, hands, handedness):
    h, w = frame.shape[:2]
    for idx, hand in enumerate(hands):
        pts = to_pixels(hand, w, h)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for i, (x, y) in enumerate(pts):
            if i in HAND_TIPS:
                cv2.circle(frame, (x, y), 7, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (x, y), 4, (255, 200, 0), -1)
        label = handedness[idx] if idx < len(handedness) else "hand"
        cv2.putText(frame, label, (pts[0][0] - 20, pts[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_face(frame, face):
    h, w = frame.shape[:2]
    pts = to_pixels(face, w, h)
    feats = face_features()
    for a, b in feats["FACEMESH_TESSELATION"]:
        if a < len(pts) and b < len(pts):
            cv2.line(frame, pts[a], pts[b], (70, 70, 70), 1)
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


def draw_clap_line(frame, hands, close: bool):
    """Join the two palms; yellow when they are close enough to be a clap."""
    if len(hands) < 2:
        return
    h, w = frame.shape[:2]
    a = g.palm_center(hands[0])
    b = g.palm_center(hands[1])
    color = (0, 255, 255) if close else (120, 120, 120)
    cv2.line(frame, (int(a[0] * w), int(a[1] * h)),
             (int(b[0] * w), int(b[1] * h)), color, 2)


def draw_text_block(frame, lines, origin=(10, 30), color=(0, 255, 255),
                    scale=0.62, thickness=2, spacing=26):
    x0, y0 = origin
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (x0, y0 + i * spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_badge(frame, text, pos, color=(0, 0, 0), bg=(0, 0, 255)):
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x - 6, y - th - 6), (x + tw + 6, y + 6), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def build_hud(fps, fps_avg, hands_count, face_on, angles, posture,
              gestures, speeds, rec, export, stats):
    """Assemble the left-hand text block."""
    lines = [
        f"fps {fps:5.1f}  avg {fps_avg:5.1f}   hands {hands_count}   "
        f"face {'on' if face_on else 'off'}",
    ]
    if rec:
        lines.append("REC")
    if export:
        lines.append("EXPORT")
    if angles:
        lines.append(
            "  ".join(f"{k} {v:5.1f}" for k, v in angles.items())
        )
    if posture:
        flags = ", ".join(posture["flags"]) or "posture ok"
        lines.append(f"posture: {flags}")
    if gestures:
        held = ", ".join(f"{hand}:{name}" for hand, name in gestures.items())
        lines.append(f"gesture: {held or '-'}")
    if speeds:
        lines.append(
            "speed: " + "  ".join(f"{h} {v:.2f}" for h, v in speeds.items())
        )
    reps = "  ".join(f"{name} {rc.count}" for name, rc in stats.reps.items())
    lines.append(f"reps: {reps}   claps {stats.claps}")
    return lines