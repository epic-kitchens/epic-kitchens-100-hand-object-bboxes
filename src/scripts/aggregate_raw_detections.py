import argparse
import os.path
import pickle
import re
import sys
from pathlib import Path
from typing import Any, List

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # isort:skip
sys.path.insert(0, project_root)  # isort:skip

from raw_detections import FrameDetections

parser = argparse.ArgumentParser(
    description="Aggregate raw detections from per-frame to per-video",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("input_dir", type=Path, help="Directory containing raw detections")
parser.add_argument(
    "output_pkl", type=Path, help="Path to write aggregate detections pickle to."
)


def main(args):
    video_id = args.input_dir.name
    if not re.match("P\d+_\d+", video_id):
        print("Input directory name must be the video ID (e.g. P01_101)")
        sys.exit(1)
    video_detections: List[FrameDetections] = [
        load_frame_detections(path) for path in get_detection_paths(args.input_dir)
    ]

    fixup_detections(video_detections, video_id)
    video_detections = sorted(video_detections, key=lambda det: det.frame_number)
    args.output_pkl.parent.mkdir(exist_ok=True, parents=True)
    print(f"Saving detections to {args.output_pkl}")
    save_video_detections(args.output_pkl, video_detections)


def fixup_detections(video_detections: List[FrameDetections], video_id: str) -> None:
    """Correct incorrect metadata in the detections"""
    for frame_detections in video_detections:
        # Video IDs were incorrectly set to the participant_id, so we need to
        # override them here.
        frame_detections.video_id = video_id


def get_detection_paths(root_dir: Path) -> List[Path]:
    def get_sort_key(p: Path) -> int:
        return int(re.match(r".*?(\d+)\.pkl$", str(p)).group(1))

    return sorted(
        [child for child in root_dir.iterdir() if child.name.endswith(".pkl")],
        key=get_sort_key,
    )


def load_frame_detections(p: Path) -> Any:
    with open(p, "rb") as f:
        return FrameDetections.from_protobuf_str(pickle.load(f))


def save_video_detections(p: Path, video_detections: List[FrameDetections]) -> None:
    with open(p, "wb") as f:
        pickle.dump(
            [det.to_protobuf().SerializeToString() for det in video_detections], f
        )


if __name__ == "__main__":
    main(parser.parse_args())
