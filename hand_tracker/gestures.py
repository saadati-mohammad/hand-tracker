"""Named gesture recognition from hand landmarks.

Gestures are classified per hand, then debounced by a hold timer so a single
pose does not emit dozens of events while it is held.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import geometry as g

GESTURES = (
    "fist",
    "open palm",
    "point",
    "peace",
    "thumbs up",
    "ok",
    "pinch",
)


@dataclass
class GestureState:
    name: str
    since: float


@dataclass
class GestureEvent:
    name: str
    hand: str
    timestamp: float


class GestureRecognizer:
    """Classifies each hand and emits events once a gesture is held."""

    def __init__(self, hold_s: float = 0.35, cooldown_s: float = 1.0):
        self.hold_s = hold_s
        self.cooldown_s = cooldown_s
        self._current: dict[str, GestureState] = {}
        self._last_fired: dict[tuple[str, str], float] = {}
        self.events: list[GestureEvent] = []

    def classify(self, hand) -> str | None:
        """Best label for one hand, or None when it does not match a gesture."""
        fingers = g.extended_fingers(hand)
        pinch = g.pinch_distance(hand)

        # OK / pinch take priority: thumb and index tips are touching, which
        # otherwise looks like a partial fist. The threshold is in palm-width
        # units, so it must be tight — a fist also brings the curled index tip
        # near the thumb without the two touching.
        if pinch < 0.5:
            others = fingers - {"thumb", "index"}
            return "ok" if not others else "pinch"

        if not fingers:
            return "fist"

        if fingers == {"thumb"}:
            # A thumbs-up also has the thumb pointing up, not sideways.
            thumb_tip = hand[g.THUMB_TIP]
            thumb_mcp = hand[2]
            if thumb_tip.y < thumb_mcp.y - 0.04:
                return "thumbs up"
            return "fist"

        if fingers == {"index"}:
            return "point"

        if fingers == {"index", "middle"}:
            return "peace"

        if len(fingers) >= 4:
            return "open palm"

        return None

    def update(self, hands, handedness, now: float | None = None) -> list[GestureEvent]:
        """Update per-hand state and return events that fired this frame."""
        now = now if now is not None else time.time()
        fired: list[GestureEvent] = []

        # Forget hands no longer in frame so held() does not report a stale
        # gesture after the hand leaves.
        present = {
            handedness[i] if i < len(handedness) else "hand"
            for i in range(len(hands))
        }
        for label in list(self._current):
            if label not in present:
                del self._current[label]

        for idx, hand in enumerate(hands):
            label = handedness[idx] if idx < len(handedness) else "hand"
            name = self.classify(hand)

            if name is None:
                self._current.pop(label, None)
                continue

            state = self._current.get(label)
            if state is None or state.name != name:
                self._current[label] = GestureState(name, now)
                continue

            held = now - state.since
            key = (label, name)
            last = self._last_fired.get(key, 0.0)
            if held >= self.hold_s and now - last >= self.cooldown_s:
                self._last_fired[key] = now
                event = GestureEvent(name=name, hand=label, timestamp=now)
                self.events.append(event)
                fired.append(event)

        return fired

    def held(self) -> dict[str, str]:
        """Currently held gesture per hand, for the HUD."""
        return {hand: state.name for hand, state in self._current.items()}

    def reset(self):
        self._current.clear()
        self._last_fired.clear()