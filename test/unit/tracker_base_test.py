import unittest

import numpy as np
from groundlight import ROI, BBoxGeometry

from test.unit.mock_provider import create_mock_sdk_provider
from tracking.tracker_base import TrackerBase


class TrackerBaseTests(unittest.TestCase):
    """
    Tests to verify that the TrackerBase code is working.
    """

    def test_reset(self):
        """
        Verifies that the TrackerBase's reset method correctly resets all internal state
        """

        tracker = TrackerBase(image_width=100, image_height=100)

        # Add some state to the tracker by creating class IDs
        tracker._get_class_id("person")
        tracker._get_class_id("car")
        tracker._get_class_id("bicycle")

        # Verify state exists before reset
        assert tracker._current_tracks is None
        assert len(tracker.class_name_to_class_id) == 3  # noqa: PLR2004
        assert tracker.class_name_to_class_id == {"person": 0, "car": 1, "bicycle": 2}

        tracker.reset()

        # Verify that _current_tracks is cleared after reset
        assert tracker._current_tracks is None, "Expected _current_tracks to be None after reset"

        # Verify that _track_id_to_class_id cache is cleared after reset
        assert len(tracker._track_id_to_class_id) == 0, "Expected _track_id_to_class_id cache to be empty after reset"
        assert not tracker._track_id_to_class_id, "Expected _track_id_to_class_id cache to be empty after reset"

        # Verify our own state is cleared
        assert len(tracker.class_name_to_class_id) == 0
        assert not tracker.class_name_to_class_id

        # Verify that after reset, new class IDs start from 0 again
        assert tracker._get_class_id("dog") == 0
        assert tracker.class_name_to_class_id == {"dog": 0}

    def test_update_raises_error(self):
        """
        Test that both update functions raise an error for TrackerBase
        """
        dataset, image_width, image_height = create_mock_sdk_provider(
            return_mode="imagequery", detections_mode="groundtruth", example_name="volleyball_example"
        )

        tracker = TrackerBase(image_width=image_width, image_height=image_height)

        for image, data, _ in dataset:
            with self.assertRaises(NotImplementedError):
                tracker.update_from_image_query(data, image)

        roi = ROI(
            label="person", score=0.9, geometry=BBoxGeometry(left=0.1, top=0.2, right=0.4, bottom=0.6, x=0.25, y=0.4)
        )
        frame_mock = np.zeros((100, 100, 3))

        with self.assertRaises(NotImplementedError):
            tracker.update_from_rois([roi], frame=frame_mock)
