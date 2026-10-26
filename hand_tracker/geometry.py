"""Landmark geometry: angles, distances, posture and gesture primitives.

All inputs are sequences of objects with normalized ``x``/``y`` (MediaPipe
landmarks). Coordinates stay normalized so thresholds do not depend on the
frame resolution.
"""

from __future__ import annotations

import math

# Pose landmark indices (MediaPipe BlazePose 33-point model)
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

# Hand landmark indices
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

FINGER_JOINTS = {
    "thumb": (2, 3, THUMB_TIP),
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}


def angle_deg(a, b, c) -> float:
    """Interior angle at ``b`` formed by ``a``-``b``-``c`` in degrees (2D)."""
    v1 = (a.x - b.x, a.y - b.y)
    v2 = (c.x - b.x, c.y - b.y)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def midpoint(a, b):
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def torso_height(pose) -> float:
    """Shoulder-to-hip distance; the natural scale for normalized thresholds."""
    shoulder = midpoint(pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER])
    hip = midpoint(pose[LEFT_HIP], pose[RIGHT_HIP])
    return math.hypot(shoulder[0] - hip[0], shoulder[1] - hip[1]) or 1e-6


def joint_angles(pose) -> dict[str, float]:
    """Elbow and knee angles for both sides."""
    return {
        "L elbow": angle_deg(pose[LEFT_SHOULDER], pose[LEFT_ELBOW], pose[LEFT_WRIST]),
        "R elbow": angle_deg(pose[RIGHT_SHOULDER], pose[RIGHT_ELBOW], pose[RIGHT_WRIST]),
        "L knee": angle_deg(pose[LEFT_HIP], pose[LEFT_KNEE], pose[LEFT_ANKLE]),
        "R knee": angle_deg(pose[RIGHT_HIP], pose[RIGHT_KNEE], pose[RIGHT_ANKLE]),
    }


def arm_raise_deg(pose, side: str) -> float:
    """Angle of the upper arm away from straight down. 0 = hanging, 180 = overhead."""
    if side == "L":
        shoulder, elbow = pose[LEFT_SHOULDER], pose[LEFT_ELBOW]
    else:
        shoulder, elbow = pose[RIGHT_SHOULDER], pose[RIGHT_ELBOW]
    dx, dy = elbow.x - shoulder.x, elbow.y - shoulder.y
    # y grows downward in image space, so straight down is (0, +1)
    return math.degrees(math.atan2(math.hypot(dx, dy), dy))


def posture_report(pose) -> dict[str, object]:
    """Coarse posture checks that are useful for a seated-desk user."""
    ls, rs = pose[LEFT_SHOULDER], pose[RIGHT_SHOULDER]
    lh, rh = pose[LEFT_HIP], pose[RIGHT_HIP]
    nose = pose[NOSE]

    shoulder_tilt = math.degrees(math.atan2(rs.y - ls.y, rs.x - ls.x))
    shoulder_width = dist(ls, rs) or 1e-6
    shoulder_mid = midpoint(ls, rs)
    # Forward slouch pulls the nose away from the shoulder midpoint in 2D.
    head_offset = abs(nose.x - shoulder_mid[0]) / shoulder_width
    hip_mid = midpoint(lh, rh)
    torso_lean = math.degrees(
        math.atan2(
            shoulder_mid[0] - hip_mid[0],
            (hip_mid[1] - shoulder_mid[1]) or 1e-6,
        )
    )

    flags = []
    if abs(shoulder_tilt) > 8:
        flags.append("uneven shoulders")
    if head_offset > 0.45:
        flags.append("head off-centre")
    if abs(torso_lean) > 12:
        flags.append("leaning")

    return {
        "shoulder_tilt": shoulder_tilt,
        "head_offset": head_offset,
        "torso_lean": torso_lean,
        "flags": flags,
    }


def finger_extended(hand, finger: str) -> bool:
    """A finger counts as extended when its tip is farther from the wrist
    than its middle joint is. Direction-agnostic, so it works for any hand
    orientation.
    """
    _, pip, tip = FINGER_JOINTS[finger]
    return dist(hand[WRIST], hand[tip]) > dist(hand[WRIST], hand[pip]) * 1.08


def extended_fingers(hand) -> set[str]:
    return {f for f in FINGER_JOINTS if finger_extended(hand, f)}


def palm_center(hand) -> tuple[float, float]:
    xs = [lm.x for lm in hand]
    ys = [lm.y for lm in hand]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def hand_scale(hand) -> float:
    """Palm width (index-MCP to pinky-MCP), used to normalize thresholds.

    Palm width is preferred over wrist-to-finger length because it stays
    roughly constant as the fingers curl, so a closed fist does not shrink
    the scale and make nearby fingertips look like a pinch.
    """
    return dist(hand[INDEX_MCP], hand[PINKY_MCP]) or 1e-6


def pinch_distance(hand) -> float:
    """Thumb-tip to index-tip distance, normalized by palm width."""
    return dist(hand[THUMB_TIP], hand[INDEX_TIP]) / hand_scale(hand)