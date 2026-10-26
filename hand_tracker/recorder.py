"""Video recording, snapshots and landmark export."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2


def _stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


class Recorder:
    """Writes annotated video and/or landmark data to an output directory.

    Video is written with mp4v, which is available in the standard OpenCV
    build; if the writer fails to open it falls back to MJPG in an .avi.
    """

    def __init__(self, outdir: str | Path = "recordings"):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self._writer: cv2.VideoWriter | None = None
        self._video_path: Path | None = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_path: Path | None = None
        self._json_path: Path | None = None
        self._json_rows: list[dict] = []
        self._size: tuple[int, int] | None = None
        self.frame_index = 0

    # -- video -----------------------------------------------------------
    @property
    def recording(self) -> bool:
        return self._writer is not None

    @property
    def video_path(self) -> Path | None:
        return self._video_path

    def start_video(self, size: tuple[int, int], fps: float = 20.0) -> Path:
        self.stop_video()
        self._size = size
        path = self.outdir / f"session-{_stamp()}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), size
        )
        if not writer.isOpened():
            writer.release()
            path = self.outdir / f"session-{_stamp()}.avi"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                max(fps, 1.0),
                size,
            )
        if not writer.isOpened():
            raise RuntimeError(
                "Could not open a video writer. Check that OpenCV has an "
                "ffmpeg build, or install opencv-python (not headless)."
            )
        self._writer = writer
        self._video_path = path
        return path

    def write_video(self, frame_bgr):
        if self._writer is not None:
            # Guard against a resolution change mid-session.
            h, w = frame_bgr.shape[:2]
            if self._size and (w, h) != self._size:
                frame_bgr = cv2.resize(frame_bgr, self._size)
            self._writer.write(frame_bgr)

    def stop_video(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # -- landmark export -------------------------------------------------
    CSV_FIELDS = [
        "frame", "time_s", "kind", "index", "label",
        "x", "y", "z", "visibility",
    ]

    def start_export(self) -> tuple[Path, Path]:
        stamp = _stamp()
        self._csv_path = self.outdir / f"landmarks-{stamp}.csv"
        self._json_path = self.outdir / f"landmarks-{stamp}.json"
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDS)
        self._csv_writer.writeheader()
        self._json_rows = []
        return self._csv_path, self._json_path

    @property
    def exporting(self) -> bool:
        return self._csv_writer is not None

    def export_frame(self, result, time_s: float):
        """Append one frame's landmarks to both export formats."""
        if self._csv_writer is None:
            return
        self.frame_index += 1
        rows = []

        if result.pose:
            for i, lm in enumerate(result.pose):
                rows.append(("pose", i, "", lm))
        for h, hand in enumerate(result.hands):
            label = (
                result.handedness[h] if h < len(result.handedness) else "hand"
            )
            for i, lm in enumerate(hand):
                rows.append(("hand", i, label, lm))
        if result.face:
            for i, lm in enumerate(result.face):
                rows.append(("face", i, "", lm))

        for kind, index, label, lm in rows:
            self._csv_writer.writerow({
                "frame": self.frame_index,
                "time_s": round(time_s, 4),
                "kind": kind,
                "index": index,
                "label": label,
                "x": round(lm.x, 6),
                "y": round(lm.y, 6),
                "z": round(getattr(lm, "z", 0.0), 6),
                "visibility": round(getattr(lm, "visibility", 0.0), 4),
            })

        self._json_rows.append({
            "frame": self.frame_index,
            "time_s": round(time_s, 4),
            "pose": [
                {"x": round(lm.x, 5), "y": round(lm.y, 5),
                 "z": round(getattr(lm, "z", 0.0), 5)}
                for lm in result.pose
            ] if result.pose else None,
            "hands": [
                {
                    "handedness": result.handedness[h]
                    if h < len(result.handedness) else "hand",
                    "landmarks": [
                        {"x": round(lm.x, 5), "y": round(lm.y, 5),
                         "z": round(getattr(lm, "z", 0.0), 5)}
                        for lm in hand
                    ],
                }
                for h, hand in enumerate(result.hands)
            ],
        })

    def stop_export(self) -> list[Path]:
        paths = []
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
            if self._csv_path:
                paths.append(self._csv_path)
        if self._json_path is not None:
            with open(self._json_path, "w", encoding="utf-8") as fh:
                json.dump(self._json_rows, fh)
            paths.append(self._json_path)
            self._json_path = None
        self._json_rows = []
        return paths

    # -- snapshots -------------------------------------------------------
    def snapshot(self, frame_bgr) -> Path:
        path = self.outdir / f"snapshot-{_stamp()}-{self.frame_index}.png"
        cv2.imwrite(str(path), frame_bgr)
        return path

    # -- teardown --------------------------------------------------------
    def close(self) -> list[Path]:
        self.stop_video()
        exported = self.stop_export()
        if self._video_path and self._video_path.exists():
            exported.append(self._video_path)
        return exported