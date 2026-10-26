"""Runtime settings and the on-screen settings menu."""

from __future__ import annotations

from dataclasses import dataclass

import cv2

TOGGLEABLE = (
    ("show_pose", "Pose skeleton"),
    ("show_hands", "Hand landmarks"),
    ("show_face", "Face mesh"),
    ("show_angles", "Joint angles"),
    ("show_posture", "Posture checks"),
    ("show_gestures", "Gesture labels"),
    ("show_speed", "Hand speed"),
    ("record_video", "Record video"),
    ("export_data", "Export landmark data"),
)


@dataclass
class Settings:
    show_pose: bool = True
    show_hands: bool = True
    show_face: bool = True
    show_angles: bool = True
    show_posture: bool = True
    show_gestures: bool = True
    show_speed: bool = True
    record_video: bool = False
    export_data: bool = False
    mirror: bool = True
    width: int | None = None
    height: int | None = None
    camera_index: int | None = None


class Menu:
    """A keyboard-driven settings overlay drawn on top of the video."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.open = False
        self.selected = 0

    def move(self, delta: int):
        self.selected = (self.selected + delta) % len(TOGGLEABLE)

    def toggle(self) -> str:
        attr, _label = TOGGLEABLE[self.selected]
        setattr(self.settings, attr, not getattr(self.settings, attr))
        return attr

    def render(self, frame):
        """Draw the menu panel. Returns the frame (modified in place)."""
        h, w = frame.shape[:2]
        panel_w, row_h = 380, 34
        panel_h = row_h * (len(TOGGLEABLE) + 2) + 20
        x0, y0 = (w - panel_w) // 2, (h - panel_h) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (0, 200, 255), 2)

        cv2.putText(frame, "Settings  (up/down, enter, esc)", (x0 + 16, y0 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

        for i, (attr, label) in enumerate(TOGGLEABLE):
            y = y0 + 44 + i * row_h
            enabled = getattr(self.settings, attr)
            if i == self.selected:
                cv2.rectangle(frame, (x0 + 6, y - 20), (x0 + panel_w - 6, y + 10),
                              (60, 60, 60), -1)
                color = (255, 255, 255)
            else:
                color = (170, 170, 170)
            cv2.putText(frame, label, (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 1)
            state = "ON" if enabled else "OFF"
            state_color = (0, 220, 0) if enabled else (90, 90, 220)
            cv2.putText(frame, state, (x0 + panel_w - 80, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

        footer = f"camera {self.settings.camera_index}   mirror {self.settings.mirror}"
        cv2.putText(frame, footer, (x0 + 16, y0 + panel_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        return frame