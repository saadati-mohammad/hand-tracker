"""Command-line entry point: python -m hand_tracker"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .app import App
from .capture import list_cameras
from .settings import Settings


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hand_tracker",
        description=(
            "Real-time full-body tracking: pose skeleton, hand landmarks, "
            "face mesh, gestures, posture, rep counting, recording and export."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--camera", type=int, metavar="N",
                   help="camera index to use (default: auto-detect)")
    p.add_argument("--width", type=int, help="requested capture width")
    p.add_argument("--height", type=int, help="requested capture height")
    p.add_argument("--outdir", default="recordings",
                   help="directory for recordings, exports and snapshots")
    p.add_argument("--no-face", action="store_true",
                   help="start with the face mesh disabled (faster)")
    p.add_argument("--no-mirror", action="store_true",
                   help="do not horizontally flip the preview")
    p.add_argument("--record", action="store_true",
                   help="start recording video immediately")
    p.add_argument("--export", action="store_true",
                   help="start landmark export (CSV + JSON) immediately")
    p.add_argument("--list-cameras", action="store_true",
                   help="print available camera indices and exit")
    return p


def settings_from_args(args) -> Settings:
    return Settings(
        show_face=not args.no_face,
        mirror=not args.no_mirror,
        record_video=args.record,
        export_data=args.export,
        width=args.width,
        height=args.height,
        camera_index=args.camera,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_cameras:
        found = list_cameras()
        if found:
            print("Available camera indices:", ", ".join(map(str, found)))
        else:
            print("No cameras found.")
        return 0

    try:
        App(settings_from_args(args), outdir=args.outdir).run()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            print(f"Error: {exc.code}", file=sys.stderr)
            return 1
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())