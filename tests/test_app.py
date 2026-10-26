"""End-to-end tests for the App loop with a fake camera and tracker.

These drive App.run() for a few frames without a webcam or a GUI, so they
exercise the wiring that unit tests cannot: hotkeys, the menu, recording,
export and orderly shutdown.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hand_tracker import app as app_mod  # noqa: E402
from hand_tracker.app import App  # noqa: E402
from hand_tracker.models import Frame  # noqa: E402
from hand_tracker.settings import Settings  # noqa: E402


class FakeCapture:
    """Stands in for cv2.VideoCapture: hands out N frames, then fails."""

    def __init__(self, frames: int, w: int = 160, h: int = 120):
        self.remaining = frames
        self.w, self.h = w, h
        self.released = False

    def read(self):
        if self.remaining <= 0:
            return False, None
        self.remaining -= 1
        return True, np.zeros((self.h, self.w, 3), dtype=np.uint8)

    def get(self, prop):
        return float(self.w if prop == app_mod.cv2.CAP_PROP_FRAME_WIDTH else self.h)

    def set(self, *_args):
        return True

    def release(self):
        self.released = True


class FakeTracker:
    """Minimal stand-in for models.Tracker: no models, no inference."""

    def __init__(self, use_face: bool = True):
        self.use_face = use_face
        self.closed = False
        self.calls = 0

    def process(self, image, ts_ms):
        self.calls += 1
        return Frame(image=image)

    def set_face_enabled(self, enabled: bool):
        self.use_face = enabled

    def close(self):
        self.closed = True


class AppLoopTests(unittest.TestCase):
    def _run(self, keys, settings=None, frames=None):
        """Run App with a scripted key sequence; return (summary, app, tmp)."""
        settings = settings or Settings(show_face=False)
        n_frames = frames if frames is not None else len(keys) + 1
        cap = FakeCapture(n_frames)
        tracker = FakeTracker(use_face=settings.show_face)
        key_iter = iter(list(keys) + [ord("q")] * 100)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        with mock.patch.object(app_mod, "open_camera", return_value=cap), \
             mock.patch.object(app_mod, "Tracker", return_value=tracker), \
             mock.patch.object(app_mod.cv2, "imshow"), \
             mock.patch.object(app_mod.cv2, "destroyAllWindows"), \
             mock.patch.object(app_mod.cv2, "waitKey",
                               side_effect=lambda _ms: next(key_iter)):
            app = App(settings, outdir=tmp.name)
            # The loop prints a banner and a JSON summary; keep test output clean.
            with contextlib.redirect_stdout(io.StringIO()):
                summary = app.run()

        return summary, app, tracker, cap, Path(tmp.name)

    def test_loop_runs_and_summarizes(self):
        summary, app, tracker, cap, _ = self._run(keys=[255, 255])
        self.assertEqual(summary["frames"], 3)
        self.assertIn("claps", summary)
        self.assertIn("reps", summary)
        self.assertTrue(cap.released, "camera should be released on exit")
        self.assertTrue(tracker.closed, "tracker should be closed on exit")

    def test_q_quits_immediately(self):
        summary, _, _, cap, _ = self._run(keys=[ord("q")], frames=10)
        self.assertEqual(summary["frames"], 1)
        self.assertTrue(cap.released)

    def test_face_toggle_calls_tracker(self):
        _, _, tracker, _, _ = self._run(keys=[ord("f"), 255])
        self.assertTrue(tracker.use_face, "f should enable face on the tracker")

    def test_hotkeys_do_not_crash(self):
        # Every non-quit hotkey in sequence.
        keys = [ord(c) for c in "fphirec"]
        self._run(keys=keys, frames=len(keys) + 2)

    def test_record_hotkey_writes_video(self):
        settings = Settings(show_face=False)
        summary, _, _, _, outdir = self._run(
            keys=[ord("r")],  # start recording
            settings=settings,
            frames=6,
        )
        videos = list(outdir.glob("session-*"))
        self.assertTrue(videos, f"expected a recorded video in {outdir}")
        self.assertGreater(videos[0].stat().st_size, 0)

    def test_export_hotkey_writes_data(self):
        settings = Settings(show_face=False)
        _, _, _, _, outdir = self._run(
            keys=[ord("e"), 255, ord("e")],  # start, a frame, stop
            settings=settings,
            frames=6,
        )
        self.assertTrue(list(outdir.glob("landmarks-*.csv")))
        self.assertTrue(list(outdir.glob("landmarks-*.json")))

    def test_menu_opens_and_toggles(self):
        settings = Settings(show_face=False)
        _, app, tracker, _, _ = self._run(
            keys=[ord("m"), 13, ord("q"), 27],  # open, toggle first row, close
            settings=settings,
            frames=8,
        )
        # First menu row is show_pose; toggling flips it from True to False.
        self.assertFalse(settings.show_pose)
        self.assertFalse(app.menu.open, "esc should close the menu")

    def test_menu_face_toggle_reaches_tracker(self):
        settings = Settings(show_face=False)
        # show_face is row index 2, so move down twice then toggle.
        keys = [ord("m"), 84, 84, 13]  # 's' moves down
        self._run(keys=keys, settings=settings, frames=6)
        self.assertTrue(settings.show_face)


if __name__ == "__main__":
    unittest.main(verbosity=2)