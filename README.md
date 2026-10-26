# hand-tracker

Real-time full-body tracking from a webcam: pose skeleton, hand landmarks,
face mesh, gesture recognition, posture checks, rep counting, hand-speed
readout, clap detection, video recording and landmark export.

## Requirements

- Python 3.10+
- A webcam (DroidCam / phone-as-webcam works — the camera scan skips
  indices that fail to open)
- The three MediaPipe `.task` models and `face_mesh_connections.json` in
  `assets/` (already in this repo)

```bash
pip install -r requirements.txt
```

Use the **full** `opencv-python` build, not `opencv-python-headless` —
the app needs `cv2.imshow` and the video writers.

## Run

```bash
python -m hand_tracker
```

Install it once to get the `hand-tracker` command:

```bash
pip install -e .
hand-tracker
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--camera N` | pick a camera index (default: auto-detect) |
| `--width W --height H` | requested capture resolution |
| `--outdir DIR` | where recordings/exports/snapshots go (default `recordings`) |
| `--no-face` | start with the face mesh off (it is the costliest model) |
| `--no-mirror` | do not flip the preview |
| `--record` | start recording immediately |
| `--export` | start CSV + JSON landmark export immediately |
| `--list-cameras` | print working camera indices and exit |

## Keys

| Key | Action |
| --- | --- |
| `q` / `Esc` | quit |
| `m` | open the settings menu (arrows move, Enter toggles) |
| `f` | toggle face mesh |
| `p` | toggle pose skeleton |
| `h` | toggle hand landmarks |
| `i` | toggle mirror |
| `c` | cycle camera |
| `r` | start/stop video recording |
| `e` | start/stop landmark export |

## What it tracks

- **Pose**: 33-point BlazePose skeleton, elbow/knee angles, torso lean,
  uneven shoulders and head-off-centre posture flags.
- **Hands**: 21 landmarks per hand; finger extension, palm centre, pinch.
- **Gestures**: fist, open palm, point, peace, thumbs up, ok, pinch —
  debounced by a hold timer and a cooldown so one pose is not counted
  dozens of times.
- **Stats**: squat and push-up rep counting with hysteresis, per-hand
  speed, clap count, gesture and posture tallies.
- **Recording/export**: annotated MP4 (falls back to MJPG/AVI), landmark
  CSV and JSON, PNG snapshots.

The session summary (frames, avg FPS, claps, gestures, reps, peak hand
speeds, posture flags) is printed as JSON on exit.

## Tests

```bash
python -m unittest discover -s tests -v
```

`tests/test_units.py` covers geometry, gesture classification, rep
counting, hand speed and the recorder's video/CSV/JSON output.
`tests/test_app.py` drives the full `App.run()` loop with a fake camera
and tracker, covering hotkeys, the settings menu, recording, export and
orderly shutdown — no webcam or GUI required.

The tests need `numpy`, so run them with the same interpreter that has the
dependencies installed.

## Layout

```
assets/            MediaPipe models + face mesh connections
hand_tracker/
  __main__.py      CLI (argparse) and entry point
  app.py           capture loop, hotkeys, menu, feature wiring
  capture.py       camera discovery and configuration
  models.py        landmarker lifecycle + per-frame inference
  geometry.py      angles, distances, posture, gesture primitives
  gestures.py      gesture classification + debouncing
  stats.py         rep counting, hand speed, session summary
  recorder.py      video, CSV/JSON export, snapshots
  render.py        skeleton/hand/face drawing and the HUD
  settings.py      runtime settings and the on-screen menu
  paths.py         asset lookup (source or PyInstaller)
tests/
  test_units.py    unit tests
```