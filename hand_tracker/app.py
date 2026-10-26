"""Main application loop wiring capture, tracking, features and rendering."""

from __future__ import annotations

import json
import math
import time

import cv2

from . import geometry as g
from . import render
from .capture import CameraError, configure, list_cameras, open_camera
from .gestures import GestureRecognizer
from .models import Tracker
from .recorder import Recorder
from .settings import Menu, Settings
from .stats import SessionStats

CLAP_CLOSE_RATIO = 0.18
CLAP_COOLDOWN_S = 0.8
WINDOW = "Hand Tracker"


class App:
    def __init__(self, settings: Settings, outdir: str = "recordings"):
        self.settings = settings
        self.menu = Menu(settings)
        self.stats = SessionStats()
        self.gestures = GestureRecognizer()
        self.recorder = Recorder(outdir)
        self.tracker: Tracker | None = None
        self.cap = None
        self._fps = 0.0
        self._prev_t = time.time()
        self._hands_together = False
        self._last_clap = 0.0

    # -- setup -----------------------------------------------------------
    def open(self):
        self.cap = open_camera(self.settings.camera_index)
        if self.settings.width and self.settings.height:
            configure(self.cap, self.settings.width, self.settings.height)
        self.tracker = Tracker(use_face=self.settings.show_face)

    def close(self):
        exported = self.recorder.close()
        if self.tracker is not None:
            self.tracker.close()
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        return exported

    # -- per-frame features ----------------------------------------------
    def _apply_toggle(self, attr: str):
        """React to a setting that changed from the menu or a hotkey."""
        if attr == "show_face" and self.tracker is not None:
            self.tracker.set_face_enabled(self.settings.show_face)
        elif attr == "record_video" and self.cap is not None:
            if self.settings.record_video:
                w, h = (
                    int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                try:
                    self.recorder.start_video((w, h), fps=20.0)
                except RuntimeError as exc:
                    print(f"Recording unavailable: {exc}")
                    self.settings.record_video = False
            else:
                self.recorder.stop_video()
        elif attr == "export_data":
            if self.settings.export_data:
                self.recorder.start_export()
            else:
                for path in self.recorder.stop_export():
                    print(f"Saved {path}")

    def _handle_key(self, key: int) -> bool:
        """Returns False when the app should quit."""
        if key == 255:
            return True

        if self.menu.open:
            if key in (ord("q"), 27):
                self.menu.open = False
            elif key in (82, ord("w")):
                self.menu.move(-1)
            elif key in (84, ord("s")):
                self.menu.move(1)
            elif key in (13, 10, ord(" ")):
                self._apply_toggle(self.menu.toggle())
            return True

        if key in (ord("q"), 27):
            return False
        if key == ord("m"):
            self.menu.open = True
        elif key == ord("f"):
            self.settings.show_face = not self.settings.show_face
            self._apply_toggle("show_face")
        elif key == ord("p"):
            self.settings.show_pose = not self.settings.show_pose
        elif key == ord("h"):
            self.settings.show_hands = not self.settings.show_hands
        elif key == ord("r"):
            self.settings.record_video = not self.settings.record_video
            self._apply_toggle("record_video")
        elif key == ord("e"):
            self.settings.export_data = not self.settings.export_data
            self._apply_toggle("export_data")
        elif key == ord("i"):
            self.settings.mirror = not self.settings.mirror
        elif key == ord("c"):
            self._cycle_camera()
        return True

    def _cycle_camera(self):
        cams = list_cameras()
        if len(cams) < 2:
            print("Only one camera available.")
            return
        current = self.settings.camera_index
        nxt = (
            cams[(cams.index(current) + 1) % len(cams)]
            if current in cams
            else cams[0]
        )
        self.settings.camera_index = nxt
        self.cap.release()
        self.cap = open_camera(nxt)
        if self.settings.width and self.settings.height:
            configure(self.cap, self.settings.width, self.settings.height)
        print(f"Switched to camera {nxt}")

    def _step(self, result, now: float):
        """Update all feature trackers for one frame."""
        s = self.settings
        stats = self.stats

        angles = g.joint_angles(result.pose) if result.pose else None
        posture = g.posture_report(result.pose) if result.pose else None

        if angles:
            for rc in stats.reps.values():
                rc.update(angles)
            if posture:
                stats.note_posture(posture["flags"])

        for ev in self.gestures.update(result.hands, result.handedness, now):
            stats.note_gesture(ev.name)

        speeds = stats.speed.update(result.hands, result.handedness, now)

        # Clap: two palms close together, edge-triggered with a cooldown.
        together = False
        if len(result.hands) == 2:
            d = math.dist(
                g.palm_center(result.hands[0]), g.palm_center(result.hands[1])
            )
            together = d < CLAP_CLOSE_RATIO
        if (
            together
            and not self._hands_together
            and now - self._last_clap > CLAP_COOLDOWN_S
        ):
            self._last_clap = now
            stats.claps += 1
            print(f"CLAP (total {stats.claps})")
        self._hands_together = together

        return angles, posture, speeds, together

    # -- main loop -------------------------------------------------------
    def run(self) -> dict:
        try:
            self.open()
        except CameraError as exc:
            raise SystemExit(str(exc))
        except FileNotFoundError as exc:
            raise SystemExit(f"Model files missing: {exc}")

        print(
            "Hand Tracker\n"
            "  q/esc quit   m menu   f face   p pose   h hands\n"
            "  r record    e export  i mirror c camera"
        )

        exported = []
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Camera frame grab failed; stopping.")
                    break

                s = self.settings
                if s.mirror:
                    frame = cv2.flip(frame, 1)
                now = time.time()
                dt = max(now - self._prev_t, 1e-6)
                self._prev_t = now
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
                self.stats.frames += 1

                result = self.tracker.process(frame, int(now * 1000))
                angles, posture, speeds, together = self._step(result, now)

                if s.export_data:
                    self.recorder.export_frame(result, now - self.stats.started)

                out = frame
                if s.show_face and result.face:
                    render.draw_face(out, result.face)
                if s.show_pose and result.pose:
                    render.draw_pose(out, result.pose)
                if s.show_hands:
                    render.draw_hands(out, result.hands, result.handedness)
                render.draw_clap_line(out, result.hands, together)

                hud = render.build_hud(
                    fps=self._fps,
                    fps_avg=self.stats.fps_avg,
                    hands_count=len(result.hands),
                    face_on=self.tracker.use_face,
                    angles=angles if s.show_angles else None,
                    posture=posture if s.show_posture else None,
                    gestures=self.gestures.held() if s.show_gestures else None,
                    speeds=speeds if s.show_speed else None,
                    rec=self.recorder.recording,
                    export=self.recorder.exporting,
                    stats=self.stats,
                )
                render.draw_text_block(out, hud)
                if self.recorder.recording:
                    render.draw_badge(out, "REC", (out.shape[1] - 70, 30))

                if s.record_video and self.recorder.recording:
                    self.recorder.write_video(out)

                if self.menu.open:
                    self.menu.render(out)

                cv2.imshow(WINDOW, out)
                if not self._handle_key(cv2.waitKey(1) & 0xFF):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            exported = self.close()
            summary = self.stats.summary()

        if exported:
            print("\nSaved files:")
            for path in exported:
                print(f"  {path}")

        print("\nSession summary:")
        print(json.dumps(summary, indent=2))
        return summary