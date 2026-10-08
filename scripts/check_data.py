"""Validate the directory layout used by the tracking lab."""

import argparse
from pathlib import Path
import sys


SEQUENCES = tuple(f"video_{index}" for index in range(1, 6))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-data-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.lab_data_root.expanduser().resolve()
    errors = []

    if not root.is_dir():
        parser.error(f"data directory does not exist: {root}")

    for sequence in SEQUENCES:
        sequence_dir = root / sequence
        image_dir = sequence_dir / "img1"
        images = sorted(path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES) if image_dir.is_dir() else []
        if not sequence_dir.is_dir():
            errors.append(f"{sequence}: missing directory {sequence_dir}")
            continue
        if not images:
            errors.append(f"{sequence}: no images found in {image_dir}")
        else:
            print(f"{sequence}: {len(images)} frames ({images[0].name} to {images[-1].name})")

        if sequence == "video_1":
            gt_file = sequence_dir / "gt" / "gt.txt"
            if gt_file.is_file():
                print(f"{sequence}: ground truth found ({gt_file})")
            else:
                errors.append(f"{sequence}: missing ground truth {gt_file}")
        elif (sequence_dir / "gt" / "gt.txt").exists():
            print(f"{sequence}: ground truth present")
        else:
            print(f"{sequence}: no ground truth (visual review only)")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Data check passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
