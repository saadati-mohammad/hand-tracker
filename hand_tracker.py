import importlib.util
import math
import subprocess
import sys
import time
from pathlib import Path

REQUIRED = {"cv2": "opencv-python", "mediapipe": "mediapipe", "numpy": "numpy"}


def ensure_dependencies():
    missing = [pkg for mod, pkg in REQUIRED.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print(f"Installing missing dependencies: {', '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])


ensure_dependencies()

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]
TIP_IDS = (4, 8, 12, 16, 20)

CLAP_CLOSE_RATIO = 0.18
CLAP_COOLDOWN_S = 0.8


def palm_center(landmarks):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def main():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera 0")

    print("Hand tracker running. Press 'q' or ESC in the video window to quit.")

    hands_were_together = False
    last_clap = 0.0
    clap_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame grab failed; stopping.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            palms = []

            for idx, landmarks in enumerate(result.hand_landmarks):
                label = "hand"
                if idx < len(result.handedness) and result.handedness[idx]:
                    label = result.handedness[idx][0].category_name

                pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
                for a, b in CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for i, (x, y) in enumerate(pts):
                    color = (0, 0, 255) if i in TIP_IDS else (255, 200, 0)
                    radius = 7 if i in TIP_IDS else 4
                    cv2.circle(frame, (x, y), radius, color, -1)
                    if i in TIP_IDS:
                        cv2.putText(frame, str(i), (x + 6, y - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                px, py = palm_center(landmarks)
                palms.append((px, py))
                cv2.putText(frame, f"{label} ({pts[0][0]},{pts[0][1]})",
                            (pts[0][0] - 30, pts[0][1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            together = False
            if len(palms) == 2:
                dist = math.dist(palms[0], palms[1])
                together = dist < CLAP_CLOSE_RATIO
                cv2.line(frame,
                         (int(palms[0][0] * w), int(palms[0][1] * h)),
                         (int(palms[1][0] * w), int(palms[1][1] * h)),
                         (0, 255, 255), 2)

            now = time.time()
            if together and not hands_were_together and now - last_clap > CLAP_COOLDOWN_S:
                clap_count += 1
                last_clap = now
                print(f"CLAP detected (total: {clap_count})")
            hands_were_together = together

            status = f"claps: {clap_count}   hands: {len(palms)}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.imshow("Hand Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()