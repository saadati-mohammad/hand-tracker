"""Locate model assets whether running from source or a PyInstaller bundle."""

import sys
from pathlib import Path


def asset_dir() -> Path:
    """Directory holding the .task models and face mesh data.

    PyInstaller unpacks bundled data into ``sys._MEIPASS``; from source the
    assets live next to the package, one level up.
    """
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled) / "assets"
    return Path(__file__).resolve().parent.parent / "assets"


def asset(name: str) -> Path:
    path = asset_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"Missing asset '{name}'. Expected it in {asset_dir()}."
        )
    return path