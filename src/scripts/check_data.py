import argparse
from pathlib import Path
from typing import List, Optional

from epic_kitchens.hoa.types import BBox, FloatVector, HandSide, HandState

from epic_kitchens.hoa import FrameDetections, HandDetection, ObjectDetection

from epic_kitchens.hoa.io import load_detections


parser = argparse.ArgumentParser(
    description="Sanity check hand-object detections",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "detections_pkl", type=Path, help="Path to hand-object detections pkl."
)
parser.add_argument(
    "-n", "--n-frames", type=int, help="Expected number of frames in video"
)


class DetectionChecker:
    def __init__(self, n_frames: Optional[int]):
        self.n_frames = n_frames

    def check(self, video_detections: List[FrameDetections]) -> None:
        if self.n_frames is not None:
            if len(video_detections) != self.n_frames:
                raise ValueError(
                    f"Expected video_detections to contain {self.n_frames} "
                    f"detections, but contained {len(video_detections)}."
                )
        for frame_detections in video_detections:
            self.check_frame_detections(frame_detections)

    def check_frame_detections(self, frame_detections: FrameDetections) -> None:
        if self.n_frames is not None:
            if not (1 <= frame_detections.frame_number <= self.n_frames):
                raise ValueError(
                    "Expected frame_detections to have frame_number "
                    f"between 1 and {self.n_frames}, but was "
                    f"{frame_detections.frame_number}"
                )
        for obj in frame_detections.objects:
            self.check_object_detection(obj)

        for hand in frame_detections.hands:
            self.check_hand_detection(hand)

    def check_object_detection(self, object_detection: ObjectDetection) -> None:
        self.check_bbox(object_detection.bbox)
        self.check_score(object_detection.score)

    def check_hand_detection(self, hand_detection: HandDetection) -> None:
        self.check_bbox(hand_detection.bbox)
        self.check_score(hand_detection.score)
        self.check_vector(hand_detection.object_offset)
        if not isinstance(hand_detection.side, HandSide):
            raise ValueError(
                f"Expected hand side to be an instance of HandSide but "
                f"was {hand_detection.side}"
            )
        if not isinstance(hand_detection.state, HandState):
            raise ValueError(
                f"Expected hand state to be an instance of HandState but "
                f"was {hand_detection.state}"
            )

    def check_score(self, score: float) -> None:
        if not (0 <= score <= 1):
            raise ValueError(f"Expected score to be between 0--1 but was {score}")

    def check_bbox(self, bbox: BBox) -> None:

        for coord in [
            "top_left_x",
            "top",
            "right",
            "bottom",
        ]:
            value = getattr(bbox, coord)
            if not (0 <= value <= 1):
                raise ValueError(f"Expected bbox {coord} ({value}) to be between 0--1.")

        if not (bbox.left <= bbox.right):
            raise ValueError(
                f"Expected bbox top_left_x ({bbox.left}) to be "
                f"less than or equal to right ({bbox.right}"
            )

        if not (bbox.top <= bbox.bottom):
            raise ValueError(
                f"Expected bbox top ({bbox.top}) to be "
                f"less than or equal to bottom ({bbox.bottom}"
            )

    def check_vector(self, vector: FloatVector) -> None:
        if not (-1 <= vector.x <= 1):
            raise ValueError(
                f"Expected vector x component to be between -1 -- 1 but was {vector.x}"
            )
        if not (-1 <= vector.y <= 1):
            raise ValueError(
                    f"Expected vector y component to be between -1 -- 1 but was"
                    f" {vector.y}"
            )


def main(args):
    detections: List[FrameDetections] = load_detections(args.detections_pkl)
    checker = DetectionChecker(n_frames=args.n_frames)
    checker.check(detections)


if __name__ == "__main__":
    main(parser.parse_args())
