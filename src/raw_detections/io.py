from pathlib import Path
from typing import List, Union

from .types import FrameDetections


def load_detections(filename: Union[str, Path]) -> List[FrameDetections]:
    import pickle

    with open(filename, "rb") as f:
        return [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]


def load_frame_detections(filename: Union[str, Path]) -> FrameDetections:
    import pickle

    with open(filename, "rb") as f:
        return FrameDetections.from_protobuf_str(pickle.load(f))


def save_detections(detections: List[FrameDetections], filepath: Union[str, Path]) -> None:
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump([d.to_protobuf().SerializeToString() for d in detections], f)