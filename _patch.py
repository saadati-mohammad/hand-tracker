from pathlib import Path

p = Path(__file__).with_name("hand_tracker.py")
src = p.read_text(encoding="utf-8")

old = "import math\nimport time\nfrom pathlib import Path\n"

new = '''import importlib.util
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
'''

if "ensure_dependencies" in src:
    print("already patched")
elif old not in src:
    print("ERROR: anchor text not found")
else:
    p.write_text(src.replace(old, new, 1), encoding="utf-8")
    print("patched OK")