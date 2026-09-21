"""One-time helper to fetch the OpenCV Zoo ONNX models this pipeline needs.

Run once before first use:

    python download_models.py

If your network blocks GitHub raw downloads, download the two files manually
from https://github.com/opencv/opencv_zoo (paths below) and place them in
models/ under the same filenames.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"

# (url, minimum expected byte size) -- the size floor catches truncated/
# interrupted downloads (e.g. a killed process) that would otherwise leave a
# corrupt file behind that looks "present" to a naive existence check.
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        200_000,
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        30_000_000,
    ),
    "emotion_ferplus.onnx": (
        "https://github.com/onnx/models/raw/main/validated/vision/"
        "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        30_000_000,
    ),
}


def main() -> int:
    MODELS_DIR.mkdir(exist_ok=True)
    for filename, (url, min_size) in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size >= min_size:
            print(f"[skip] {filename} already present")
            continue
        if dest.exists():
            print(f"[refetch] {filename} looked incomplete ({dest.stat().st_size} bytes) -- redownloading")

        print(f"[download] {filename} <- {url}")
        # Download to a temp file first and only rename on success, so a
        # cancelled/failed download never leaves a broken file under the
        # final name that a later run would mistake for "already present".
        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, tmp_dest)
            if tmp_dest.stat().st_size < min_size:
                raise IOError(f"downloaded file is only {tmp_dest.stat().st_size} bytes (expected >= {min_size})")
            tmp_dest.replace(dest)
        except Exception as exc:  # surface any network/model error to the user
            print(f"[FATAL] Could not download {filename}: {exc}")
            print("Download it manually from the OpenCV Zoo and place it in models/.")
            tmp_dest.unlink(missing_ok=True)
            return 1
    print("All models present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
