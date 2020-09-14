import argparse
import os.path
import pickle
import sys
from pathlib import Path
from typing import List

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # isort:skip
sys.path.insert(0, project_root)  # isort:skip
sys.path.insert(0, os.path.join(project_root, 'public_lib'))  # isort:skip

import numpy as np

from epic_kitchens.hoa.types import BBox as ReleasableBBox
from epic_kitchens.hoa.types import FloatVector
from epic_kitchens.hoa.types import FrameDetections as ReleasableFrameDetections
from epic_kitchens.hoa.types import HandDetection as ReleasableHandDetection
from epic_kitchens.hoa.types import HandSide as ReleasableHandSide
from epic_kitchens.hoa.types import HandState as ReleasableHandState
from epic_kitchens.hoa.types import ObjectDetection as ReleasableObjectDetection
from raw_detections.types import BBox as RawBBox
from raw_detections.types import FrameDetections as RawFrameDetections
from raw_detections.types import HandDetection as RawHandDetection
from raw_detections.types import HandSide as RawHandSide
from raw_detections.types import HandState as RawHandState
from raw_detections.types import ObjectDetection as RawObjectDetection
from raw_detections.types import OffsetVector


parser = argparse.ArgumentParser(
    description="Convert raw detections to releasable ones",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "raw_video_annotations_pkl",
    type=Path,
    help="Path to file containing a list of pickled raw FrameDetection protobuf "
    "strings",
)
parser.add_argument(
    "releasable_video_annotations_pkl",
    type=Path,
    help="Path to file containing a list of pickled releasable FrameDetection "
    "protobuf strings",
)
parser.add_argument(
    "--frame-height", type=int, default=256, help="Height of frame detector was run on"
)
parser.add_argument(
    "--frame-width", type=int, default=456, help="Width of frame detector was run on"
)


def main(args):
    raw_video_annotations = load_raw_video_detections(args.raw_video_annotations_pkl)
    converter = Converter(frame_height=args.frame_height, frame_width=args.frame_width)
    releasable_video_annotations = converter.convert_video_annotations(
        raw_video_annotations
    )
    save_releasable_video_detections(
        args.releasable_video_annotations_pkl, releasable_video_annotations
    )


class Converter:
    def __init__(self, frame_height: int, frame_width: int):
        self.frame_height = frame_height
        self.frame_width = frame_width

    def convert_video_annotations(
        self, raw_video_annotations: List[RawFrameDetections],
    ) -> List[ReleasableFrameDetections]:
        return [
            self.convert_frame_annotations(frame_annotations)
            for frame_annotations in raw_video_annotations
        ]

    def convert_frame_annotations(
        self, frame_annotations: RawFrameDetections,
    ) -> ReleasableFrameDetections:
        releasable = ReleasableFrameDetections(
            video_id=frame_annotations.video_id,
            frame_number=frame_annotations.frame_number,
            objects=[self.convert_object(o) for o in frame_annotations.objects],
            hands=[self.convert_hand(h) for h in frame_annotations.hands],
        )
        return releasable

    def convert_object(
        self, object_detection: RawObjectDetection
    ) -> ReleasableObjectDetection:
        return ReleasableObjectDetection(
            bbox=self.convert_bbox(object_detection.bbox), score=object_detection.score
        )

    def convert_hand(self, hand: RawHandDetection) -> ReleasableHandDetection:
        return ReleasableHandDetection(
            bbox=self.convert_bbox(hand.bbox),
            score=hand.score,
            state=self.convert_hand_state(hand.state),
            side=self.convert_hand_side(hand.side),
            object_offset=self.convert_object_offset(hand.offset),
        )

    def convert_bbox(self, bbox: RawBBox) -> ReleasableBBox:
        def clip(x: float) -> float:
            return float(np.clip(x, 0, 1))

        left = clip(bbox.top_left.x / self.frame_width)
        top = clip(bbox.top_left.y / self.frame_height)
        right = clip((bbox.top_left.x + bbox.width) / self.frame_width)
        bottom = clip((bbox.top_left.y + bbox.height) / self.frame_height)

        new_bbox = ReleasableBBox(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
        )
        return new_bbox

    def convert_hand_state(self, state: RawHandState) -> ReleasableHandState:
        return ReleasableHandState(state.value)

    def convert_hand_side(self, side: RawHandSide) -> ReleasableHandSide:
        return ReleasableHandSide(side.value)

    def convert_object_offset(self, offset: OffsetVector) -> FloatVector:
        def clip(x: float) -> float:
            return float(np.clip(x, -1, 1))

        x = clip((offset.x * offset.magnitude) / self.frame_width)
        y = clip((offset.y * offset.magnitude) / self.frame_height)

        return FloatVector(x=x, y=y)


def load_raw_video_detections(p: Path) -> List[RawFrameDetections]:
    with open(p, "rb") as f:
        return [RawFrameDetections.from_protobuf_str(s) for s in pickle.load(f)]


def save_releasable_video_detections(
    p: Path, video_detections: List[ReleasableFrameDetections]
) -> None:
    with open(p, "wb") as f:
        pickle.dump(
            [det.to_protobuf().SerializeToString() for det in video_detections], f
        )


if __name__ == "__main__":
    main(parser.parse_args())
