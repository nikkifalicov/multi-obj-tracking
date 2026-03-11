import unittest

import numpy as np
import pytest
from groundlight import ROI, BBoxGeometry
from parameterized import parameterized

# pylint: disable=unused-import
# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
from tracking import *  # noqa: F403

# pylint: enable=unused-wildcard-import
# pylint: enable=wildcard-import
# pylint: enable=unused-import
from tracking.tracker_base import TrackerBase


def _get_all_subclasses(cls):
    """
    Gets all subclasses of the provided class
    """
    subclasses = set(cls.__subclasses__())
    for subclass in cls.__subclasses__():
        subclasses.update(_get_all_subclasses(subclass))
    return subclasses


def initialize_trackers(cls, classes_to_track=None):
    """
    Takes in a class, and returns a list of initialized trackers for the provided class and all subclasses
    """
    # get all subclasses
    subclasses = _get_all_subclasses(cls)

    # instantiate the trackers
    trackers_list = [
        (subclass.__name__, subclass(image_width=100, image_height=100, classes_to_track=classes_to_track))
        for subclass in subclasses
    ]

    # add in the tracker for the class that the method was called on
    trackers_list.append((cls.__name__, cls(image_width=100, image_height=100, classes_to_track=classes_to_track)))
    return trackers_list


class TrackerCommonTests(unittest.TestCase):
    """
    Tests to verify that the common tracker code is working across the base class and all subclasses.
    """

    @parameterized.expand(initialize_trackers(TrackerBase))
    @pytest.mark.wantsgpu
    def test_base_tracker_class_ids(self, name, tracker):  # pylint: disable=unused-argument
        """
        Verifies that TrackerBase's _get_class_id method correctly handles adding and caching new classes at
        runtime
        """

        # try adding three classes
        assert tracker._get_class_id(class_name="0") == 0
        assert tracker._get_class_id(class_name="1") == 1
        assert tracker._get_class_id(class_name="2") == 2  # noqa: PLR2004

        # verify that the cache is updated
        assert tracker.class_name_to_class_id == {
            "0": 0,
            "1": 1,
            "2": 2,
        }

        # accessing an existing class id returns the same id without updating the cache
        assert tracker._get_class_id(class_name="0") == 0
        assert tracker.class_name_to_class_id == {
            "0": 0,
            "1": 1,
            "2": 2,
        }

    @parameterized.expand(initialize_trackers(TrackerBase, ["0", "1"]))
    @pytest.mark.wantsgpu
    def test_filter_to_classes_to_track(self, name, tracker):  # pylint: disable=unused-argument
        """
        Verifies that TrackerBase's _filter_to_classes_to_track method correctly filters ROIs by class.
        """

        # Create some ROIs
        rois = []
        classes = ["0", "1", "2", "3"]
        for class_name in classes:
            rois.append(
                ROI(label=class_name, score=1.0, geometry=BBoxGeometry(left=0, top=0, right=1, bottom=1, x=0.5, y=0.5))
            )

        # Filter the ROIs
        filtered_rois = tracker._filter_to_classes_to_track(rois)  # noqa: PLR2004

        # verify that the filtered ROIs are only the ones for classes 0 and 1
        assert len(filtered_rois) == 2  # noqa: PLR2004
        assert filtered_rois[0].label == "0"
        assert filtered_rois[1].label == "1"

    @parameterized.expand(initialize_trackers(TrackerBase))
    @pytest.mark.wantsgpu
    def test_rois_to_detections_basic_cases(self, name, tracker):  # pylint: disable=unused-argument
        """
        Test basic cases for _rois_to_detections method including empty list, missing dimensions,
        single ROI, and multiple ROIs.
        """

        empty_detections = tracker._rois_to_detections([])
        assert empty_detections.is_empty()
        assert len(empty_detections) == 0

        # Test single ROI conversion
        single_roi = [
            ROI(
                label="person",
                score=0.9,
                geometry=BBoxGeometry(left=0.1, top=0.2, right=0.8, bottom=0.9, x=0.45, y=0.55),
            )
        ]
        detections = tracker._rois_to_detections(single_roi)

        assert len(detections) == 1
        assert detections.xyxy.shape == (1, 4)
        assert detections.confidence == np.array([0.9])
        assert detections.class_id == np.array([0])

        # Verify coordinate conversion (0.1*100, 0.2*100, 0.8*100, 0.9*100)
        expected_xyxy = [10.0, 20.0, 80.0, 90.0]
        assert list(detections.xyxy[0]) == expected_xyxy

        # Test multiple ROIs
        multiple_rois = [
            ROI(
                label="person", score=0.9, geometry=BBoxGeometry(left=0.1, top=0.2, right=0.5, bottom=0.6, x=0.3, y=0.4)
            ),
            ROI(
                label="car", score=0.7, geometry=BBoxGeometry(left=0.6, top=0.3, right=0.9, bottom=0.8, x=0.75, y=0.55)
            ),
            ROI(
                label="person",
                score=0.85,
                geometry=BBoxGeometry(left=0.2, top=0.1, right=0.4, bottom=0.3, x=0.3, y=0.2),
            ),
        ]

        detections = tracker._rois_to_detections(multiple_rois)

        assert len(detections) == len(multiple_rois)
        assert detections.xyxy.shape == (3, 4)

        assert detections.confidence is not None
        assert detections.class_id is not None

        assert detections.confidence.shape == (3,)
        assert detections.class_id.shape == (3,)

        # Verify all confidences
        expected_confidences = [0.9, 0.7, 0.85]
        assert list(detections.confidence) == expected_confidences

        # Verify class IDs (person=0, car=1, person=0)
        expected_class_ids = [0, 1, 0]
        assert list(detections.class_id) == expected_class_ids
