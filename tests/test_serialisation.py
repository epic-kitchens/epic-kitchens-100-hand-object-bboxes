from numpy.ma.testutils import assert_close

from epic_kitchens.hoa import load_detections, save_detections

from epic_kitchens.hoa.types import (
    FrameDetections,
    ObjectDetection,
    HandDetection,
    FloatVector,
    HandState,
    HandSide,
    BBox,
)


def assert_bbox_close(expected_bbox: BBox, actual_bbox: BBox):
    assert_close(expected_bbox.left, actual_bbox.left)
    assert_close(expected_bbox.top, actual_bbox.top)
    assert_close(expected_bbox.right, actual_bbox.right)
    assert_close(expected_bbox.bottom, actual_bbox.bottom)


def assert_float_vector_close(
    expected_float_vector: FloatVector, actual_float_vector: FloatVector
):
    assert_close(expected_float_vector.x, actual_float_vector.x)
    assert_close(expected_float_vector.y, actual_float_vector.y)


def test_serialisation_round_trip_is_idempotent(tmp_path):
    video_id = "P01_101"
    frame_number = 10

    detections = FrameDetections(
        video_id=video_id,
        frame_number=frame_number,
        objects=[ObjectDetection(bbox=BBox(0.1, 0.2, 0.3, 0.4), score=0.1)],
        hands=[
            HandDetection(
                bbox=BBox(0.2, 0.3, 0.4, 0.5),
                score=0.2,
                state=HandState.PORTABLE_OBJECT,
                side=HandSide.RIGHT,
                object_offset=FloatVector(x=0.1, y=0.1),
            )
        ],
    )

    filepath = tmp_path / (video_id + ".pkl")

    save_detections([detections], filepath)
    loaded_detections = load_detections(filepath)[0]

    assert detections.video_id == loaded_detections.video_id
    assert detections.frame_number == loaded_detections.frame_number

    assert_close(detections.objects[0].score, loaded_detections.objects[0].score)
    assert_bbox_close(detections.objects[0].bbox, loaded_detections.objects[0].bbox)

    assert_close(detections.hands[0].score, loaded_detections.hands[0].score)
    assert detections.hands[0].side == loaded_detections.hands[0].side
    assert detections.hands[0].state == loaded_detections.hands[0].state
    assert_float_vector_close(
        detections.hands[0].object_offset, loaded_detections.hands[0].object_offset
    )
    assert_bbox_close(detections.hands[0].bbox, loaded_detections.hands[0].bbox)
