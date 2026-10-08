"""Run YOLO tracking on a directory of ordered image frames."""

import argparse
from itertools import islice
from pathlib import Path
import tempfile

import cv2
from ultralytics import YOLO


TRACKERS = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
    "botsort-reid": "botsort.yaml",
    "ocsort": "ocsort.yaml",
    "deepocsort": "deepocsort.yaml",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def tracker_config(name, work_dir):
    if name != "botsort-reid":
        return TRACKERS[name]

    from ultralytics.utils import YAML
    from ultralytics.utils.checks import check_yaml

    config = YAML.load(check_yaml("botsort.yaml"))
    config["with_reid"] = True
    config["model"] = "auto"
    path = work_dir / "botsort-reid.yaml"
    YAML.save(path, config)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Directory containing image frames")
    parser.add_argument("--seq-name", required=True)
    parser.add_argument("--tracker", choices=TRACKERS, default="bytetrack")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("runs/lab"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    frames = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES) if source.is_dir() else []
    if not frames:
        parser.error(f"no image frames found in {source}")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        parser.error("--conf and --iou must be between 0 and 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")

    output = args.out.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    track_file = output / f"{args.seq_name}.txt"
    video_writer = None

    with tempfile.TemporaryDirectory(prefix="trackeval-") as temp_dir:
        temp_path = Path(temp_dir)
        tracker = tracker_config(args.tracker, temp_path)
        model = YOLO(args.model)
        stream = model.track(
            source=str(source),
            stream=True,
            persist=True,
            tracker=tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=[0],
            device=args.device,
            verbose=False,
        )

        row_count = 0
        with track_file.open("w", encoding="utf-8") as result_file:
            frames_to_process = islice(stream, args.max_frames) if args.max_frames is not None else stream
            for frame_number, result in enumerate(frames_to_process, start=1):
                if args.save_video:
                    annotated = result.plot()
                    if video_writer is None:
                        height, width = annotated.shape[:2]
                        video_path = output / f"{args.seq_name}_preview.mp4"
                        video_writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            30.0,
                            (width, height),
                        )
                        if not video_writer.isOpened():
                            raise RuntimeError(f"could not create preview video: {video_path}")
                    video_writer.write(annotated)

                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    xywh = boxes.xywh.cpu().tolist()
                    ids = boxes.id.int().cpu().tolist()
                    scores = boxes.conf.cpu().tolist()
                    for (center_x, center_y, width, height), track_id, score in zip(xywh, ids, scores):
                        left = center_x - width / 2
                        top = center_y - height / 2
                        result_file.write(
                            f"{frame_number},{track_id},{left:.2f},{top:.2f},"
                            f"{width:.2f},{height:.2f},{score:.5f},-1,-1,-1\n"
                        )
                        row_count += 1

                if frame_number % 100 == 0:
                    print(f"{args.seq_name}: processed {frame_number}/{len(frames)} frames")

        if video_writer is not None:
            video_writer.release()

    print(f"Saved {row_count} person tracks across {min(len(frames), args.max_frames or len(frames))} frames to {track_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
