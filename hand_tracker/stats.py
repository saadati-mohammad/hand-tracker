"""Session statistics: rep counting, hand speed, and a running summary."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from . import geometry as g


@dataclass
class RepCounter:
    """Counts repetitions of a joint flexing and extending.

    A rep is one full down-up cycle of a joint angle. Two thresholds with a
    hysteresis band keep a joint parked near a boundary from chattering.
    """

    name: str
    joint: str
    down_angle: float      # at or below this the joint counts as "down"
    up_angle: float        # at or above this it counts as "up"
    count: int = 0
    _state: str = "up"

    def update(self, angles: dict[str, float]) -> bool:
        angle = angles.get(self.joint)
        if angle is None:
            return False
        if self._state == "up" and angle <= self.down_angle:
            self._state = "down"
        elif self._state == "down" and angle >= self.up_angle:
            self._state = "up"
            self.count += 1
            return True
        return False

    def reset(self):
        self.count = 0
        self._state = "up"


class SpeedTracker:
    """Normalized hand speed per hand, smoothed over a short window."""

    def __init__(self, smoothing: float = 0.6):
        self.smoothing = smoothing
        self._last: dict[str, tuple[float, float, float]] = {}
        self.speeds: dict[str, float] = {}
        self.peak: dict[str, float] = {}

    def update(self, hands, handedness, now: float) -> dict[str, float]:
        for idx, hand in enumerate(hands):
            label = handedness[idx] if idx < len(handedness) else "hand"
            cx, cy = g.palm_center(hand)
            prev = self._last.get(label)
            self._last[label] = (cx, cy, now)
            if prev is None:
                continue
            dt = now - prev[2]
            if dt <= 0:
                continue
            # Normalized units per second, so it is resolution independent.
            inst = math.hypot(cx - prev[0], cy - prev[1]) / dt
            smoothed = (
                self.smoothing * self.speeds.get(label, inst)
                + (1 - self.smoothing) * inst
            )
            self.speeds[label] = smoothed
            self.peak[label] = max(self.peak.get(label, 0.0), smoothed)
        return self.speeds

    def reset(self):
        self._last.clear()
        self.speeds.clear()
        self.peak.clear()


@dataclass
class SessionStats:
    started: float = field(default_factory=time.time)
    frames: int = 0
    claps: int = 0
    gesture_counts: dict[str, int] = field(default_factory=dict)
    reps: dict[str, RepCounter] = field(default_factory=dict)
    speed: SpeedTracker = field(default_factory=SpeedTracker)
    posture_flags_seen: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.reps:
            self.reps = {
                "squat": RepCounter(
                    name="squat", joint="L knee", down_angle=110, up_angle=155
                ),
                "pushup": RepCounter(
                    name="pushup", joint="L elbow", down_angle=100, up_angle=155
                ),
            }

    @property
    def elapsed(self) -> float:
        return max(time.time() - self.started, 1e-6)

    @property
    def fps_avg(self) -> float:
        return self.frames / self.elapsed

    def note_gesture(self, name: str):
        self.gesture_counts[name] = self.gesture_counts.get(name, 0) + 1

    def note_posture(self, flags):
        for flag in flags:
            self.posture_flags_seen[flag] = self.posture_flags_seen.get(flag, 0) + 1

    def summary(self) -> dict:
        return {
            "elapsed_s": round(self.elapsed, 2),
            "frames": self.frames,
            "avg_fps": round(self.fps_avg, 1),
            "claps": self.claps,
            "gestures": dict(self.gesture_counts),
            "reps": {name: rc.count for name, rc in self.reps.items()},
            "peak_hand_speed": {
                hand: round(v, 2) for hand, v in self.speed.peak.items()
            },
            "posture_flags": dict(self.posture_flags_seen),
        }