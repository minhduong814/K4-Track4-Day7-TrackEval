"""Evaluate a lab tracker on video_1, the only sequence with ground truth."""

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import trackeval  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab-data-root", required=True, type=Path)
    parser.add_argument("--tracker-dir", required=True, type=Path, help="Directory containing video_1.txt")
    parser.add_argument("--tracker-name", default=None)
    parser.add_argument("--seq-name", default="video_1")
    parser.add_argument("--seq-length", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.seq_length <= 600:
        parser.error("--seq-length must be between 1 and 600 for this lab dataset")

    data_root = args.lab_data_root.expanduser().resolve()
    tracker_dir = args.tracker_dir.expanduser().resolve()
    gt_file = data_root / args.seq_name / "gt" / "gt.txt"
    tracker_file = tracker_dir / f"{args.seq_name}.txt"
    if not gt_file.is_file():
        parser.error(f"ground truth not found: {gt_file}")
    if not tracker_file.is_file():
        parser.error(f"tracker result not found: {tracker_file}")

    tracker_name = args.tracker_name or tracker_dir.name
    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update({
        "USE_PARALLEL": False,
        "PRINT_ONLY_COMBINED": True,
        "PLOT_CURVES": False,
    })
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    gt_root = data_root
    temp_gt = tempfile.TemporaryDirectory(prefix="trackeval-lab-gt-") if args.seq_length < 600 else None
    if temp_gt is not None:
        gt_root = Path(temp_gt.name)
        trimmed_gt = gt_root / args.seq_name / "gt" / "gt.txt"
        trimmed_gt.parent.mkdir(parents=True)
        with gt_file.open(encoding="utf-8") as source, trimmed_gt.open("w", encoding="utf-8") as target:
            for line in source:
                fields = line.split(",", 1)
                if fields and fields[0].strip().isdigit() and int(fields[0]) <= args.seq_length:
                    target.write(line)

    dataset_config.update({
        "GT_FOLDER": str(gt_root),
        "TRACKERS_FOLDER": str(tracker_dir.parent),
        "TRACKERS_TO_EVAL": [tracker_name],
        "TRACKER_SUB_FOLDER": "",
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "SKIP_SPLIT_FOL": True,
        "SEQ_INFO": {args.seq_name: args.seq_length},
        "DO_PREPROC": True,
        "OUTPUT_FOLDER": str(tracker_dir),
    })
    evaluator = trackeval.Evaluator(evaluator_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    try:
        evaluator.evaluate([dataset], metrics)
    finally:
        if temp_gt is not None:
            temp_gt.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
