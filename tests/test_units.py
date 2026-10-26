"""Unit tests for geometry, gestures, stats and recorder.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hand_tracker import geometry as g  # noqa: E402
from hand_tracker.gestures import GestureRecognizer  # noqa: E402
from hand_tracker.recorder import Recorder  # noqa: E402
from hand_tracker.stats import RepCounter, SessionStats, SpeedTracker  # noqa: E402


class LM:
    """Minimal stand-in for a MediaPipe normalized landmark."""

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z
        self.visibility = 1.0


def pose_with(knee_angle_deg=170.0, tilt=0.0):
    """Build a 33-landmark pose whose left knee has the requested angle."""
    pose = [LM(0.5, 0.2) for _ in range(33)]
    hip = (0.5, 0.5)
    pose[g.LEFT_HIP] = LM(*hip)
    pose[g.RIGHT_HIP] = LM(hip[0] + 0.1, hip[1])
    pose[g.LEFT_SHOULDER] = LM(0.5, 0.3)
    pose[g.RIGHT_SHOULDER] = LM(0.6, 0.3 + tilt)
    pose[g.NOSE] = LM(0.55, 0.2)

    # Place the knee one upper-leg length below the hip, then swing the
    # lower leg so the interior knee angle is exactly knee_angle_deg.
    upper = 0.2
    knee = (hip[0], hip[1] + upper)
    lower_dir = math.radians(180.0 - knee_angle_deg)
    ankle = (
        knee[0] + upper * math.sin(lower_dir),
        knee[1] + upper * math.cos(lower_dir),
    )
    pose[g.LEFT_KNEE] = LM(*knee)
    pose[g.LEFT_ANKLE] = LM(*ankle)
    pose[g.RIGHT_KNEE] = LM(hip[0] + 0.1, hip[1] + 0.2)
    pose[g.RIGHT_ANKLE] = LM(hip[0] + 0.1, hip[1] + 0.4)
    pose[g.LEFT_ELBOW] = LM(0.45, 0.4)
    pose[g.LEFT_WRIST] = LM(0.45, 0.5)
    pose[g.RIGHT_ELBOW] = LM(0.65, 0.4)
    pose[g.RIGHT_WRIST] = LM(0.65, 0.5)
    return pose


def hand_with(fingers, pinch=0.9, thumb_up=False):
    """Build a 21-landmark hand with the requested fingers extended.

    Curled fingertips are kept well clear of the thumb tip so a plain fist
    does not accidentally read as an "ok" pinch.
    """
    hand = [LM(0.5, 0.7) for _ in range(21)]
    for pos, (name, (mcp, pip, tip)) in enumerate(g.FINGER_JOINTS.items()):
        base_x = 0.45 + 0.025 * pos
        hand[mcp] = LM(base_x, 0.6)
        hand[pip] = LM(base_x, 0.52)
        extended = name in fingers
        # Curled tips sit below the MCP row, close to the palm, so they do
        # not line up with the thumb tip.
        hand[tip] = LM(base_x, 0.44 if extended else 0.62)
    # Thumb is always positioned explicitly: tucked beside the curled index
    # for a fist, raised when extended.
    if "thumb" in fingers:
        hand[g.THUMB_TIP] = LM(0.45, 0.40 if thumb_up else 0.55)
    else:
        hand[g.THUMB_TIP] = LM(0.40, 0.62)
    hand[2] = LM(0.43, 0.62)
    hand[3] = LM(0.42, 0.60)
    if pinch < 0.35:
        hand[g.INDEX_TIP] = LM(hand[g.THUMB_TIP].x + 0.005,
                               hand[g.THUMB_TIP].y + 0.005)
        hand[g.INDEX_PIP] = LM(0.50, 0.52)
    return hand


class GeometryTests(unittest.TestCase):
    def test_angle_right_angle(self):
        self.assertAlmostEqual(
            g.angle_deg(LM(0, 0), LM(1, 0), LM(1, 1)), 90.0, places=1
        )

    def test_angle_straight(self):
        self.assertAlmostEqual(
            g.angle_deg(LM(0, 0), LM(1, 0), LM(2, 0)), 180.0, places=1
        )

    def test_angle_degenerate_returns_zero(self):
        a = LM(0.5, 0.5)
        self.assertEqual(g.angle_deg(a, a, a), 0.0)

    def test_joint_angles_reports_four_joints(self):
        self.assertEqual(
            set(g.joint_angles(pose_with(170))),
            {"L elbow", "R elbow", "L knee", "R knee"},
        )

    def test_constructed_knee_angle_is_recovered(self):
        for target in (100.0, 130.0, 165.0):
            angles = g.joint_angles(pose_with(target))
            self.assertAlmostEqual(angles["L knee"], target, delta=1.0)

    def test_posture_clean_pose_has_no_flags(self):
        self.assertEqual(g.posture_report(pose_with(170))["flags"], [])

    def test_posture_detects_uneven_shoulders(self):
        report = g.posture_report(pose_with(170, tilt=0.12))
        self.assertIn("uneven shoulders", report["flags"])

    def test_torso_height_positive(self):
        self.assertGreater(g.torso_height(pose_with(170)), 0)

    def test_extended_fingers(self):
        self.assertEqual(
            g.extended_fingers(hand_with({"index", "middle"})),
            {"index", "middle"},
        )

    def test_pinch_distance_small_when_touching(self):
        self.assertLess(g.pinch_distance(hand_with({"thumb", "index"}, pinch=0.1)), 0.35)

    def test_palm_center_inside_hand(self):
        cx, cy = g.palm_center(hand_with(set()))
        self.assertTrue(0.4 < cx < 0.7)
        self.assertTrue(0.4 < cy < 0.8)


class GestureTests(unittest.TestCase):
    def setUp(self):
        self.rec = GestureRecognizer(hold_s=0.0, cooldown_s=0.0)

    def test_fist(self):
        self.assertEqual(self.rec.classify(hand_with(set())), "fist")

    def test_open_palm(self):
        hand = hand_with({"index", "middle", "ring", "pinky"})
        self.assertEqual(self.rec.classify(hand), "open palm")

    def test_point(self):
        self.assertEqual(self.rec.classify(hand_with({"index"})), "point")

    def test_peace(self):
        self.assertEqual(
            self.rec.classify(hand_with({"index", "middle"})), "peace"
        )

    def test_thumbs_up(self):
        hand = hand_with({"thumb"}, thumb_up=True)
        self.assertEqual(self.rec.classify(hand), "thumbs up")

    def test_event_fires_after_hold(self):
        hand = hand_with({"index", "middle"})
        self.assertEqual(self.rec.update([hand], ["Left"], now=100.0), [])
        events = self.rec.update([hand], ["Left"], now=100.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].name, "peace")

    def test_cooldown_suppresses_repeat_events(self):
        rec = GestureRecognizer(hold_s=0.0, cooldown_s=5.0)
        hand = hand_with({"index"})
        rec.update([hand], ["Left"], now=0.0)
        rec.update([hand], ["Left"], now=0.1)
        self.assertEqual(rec.update([hand], ["Left"], now=0.2), [])

    def test_held_reports_current(self):
        self.rec.update([hand_with({"index"})], ["Right"], now=1.0)
        self.assertEqual(self.rec.held(), {"Right": "point"})

    def test_held_clears_when_hand_gone(self):
        self.rec.update([hand_with({"index"})], ["Right"], now=1.0)
        self.rec.update([], [], now=1.1)
        self.assertEqual(self.rec.held(), {})


class StatsTests(unittest.TestCase):
    def test_rep_counter_counts_cycle(self):
        rc = RepCounter("squat", "L knee", down_angle=110, up_angle=155)
        self.assertFalse(rc.update({"L knee": 170}))
        self.assertFalse(rc.update({"L knee": 100}))
        self.assertTrue(rc.update({"L knee": 160}))
        self.assertEqual(rc.count, 1)

    def test_rep_counter_no_double_count_at_boundary(self):
        rc = RepCounter("squat", "L knee", down_angle=110, up_angle=155)
        rc.update({"L knee": 100})
        rc.update({"L knee": 160})
        for _ in range(5):
            rc.update({"L knee": 158})
        self.assertEqual(rc.count, 1)

    def test_rep_counter_missing_joint_is_safe(self):
        rc = RepCounter("squat", "L knee", 110, 155)
        self.assertFalse(rc.update({"R knee": 90}))

    def test_speed_tracker_measures_motion(self):
        st = SpeedTracker(smoothing=0.0)
        h1 = hand_with(set())
        st.update([h1], ["Left"], now=0.0)
        h2 = [LM(lm.x + 0.1, lm.y) for lm in h1]
        self.assertGreater(st.update([h2], ["Left"], now=0.1)["Left"], 0.5)

    def test_speed_tracker_needs_two_samples(self):
        st = SpeedTracker()
        self.assertEqual(st.update([hand_with(set())], ["Left"], now=0.0), {})

    def test_session_summary_shape(self):
        s = SessionStats()
        s.frames = 10
        s.claps = 3
        s.note_gesture("peace")
        s.note_gesture("peace")
        s.note_posture(["leaning"])
        summary = s.summary()
        self.assertEqual(summary["claps"], 3)
        self.assertEqual(summary["gestures"]["peace"], 2)
        self.assertEqual(summary["posture_flags"]["leaning"], 1)
        self.assertIn("squat", summary["reps"])


class RecorderTests(unittest.TestCase):
    def test_video_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(tmp)
            path = rec.start_video((64, 48), fps=10)
            self.assertTrue(rec.recording)
            for _ in range(5):
                rec.write_video(np.zeros((48, 64, 3), dtype=np.uint8))
            rec.stop_video()
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            self.assertFalse(rec.recording)

    def test_export_writes_csv_and_json(self):
        import csv
        import json

        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(tmp)
            csv_path, json_path = rec.start_export()
            self.assertTrue(rec.exporting)

            class Result:
                pose = [LM(0.1, 0.2), LM(0.3, 0.4)]
                hands = [[LM(0.5, 0.6)]]
                handedness = ["Left"]
                face = None

            rec.export_frame(Result(), 0.5)
            paths = rec.stop_export()
            self.assertIn(csv_path, paths)
            self.assertIn(json_path, paths)

            with open(csv_path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 3)
            self.assertEqual({r["kind"] for r in rows}, {"pose", "hand"})

            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(len(data), 1)
            self.assertEqual(len(data[0]["pose"]), 2)
            self.assertEqual(data[0]["hands"][0]["handedness"], "Left")

    def test_snapshot_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder(tmp)
            path = rec.snapshot(np.zeros((10, 10, 3), dtype=np.uint8))
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)