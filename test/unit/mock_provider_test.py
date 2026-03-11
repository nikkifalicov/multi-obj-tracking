import time
import unittest

from groundlight.client import ROI, ImageQuery
from PIL import Image

from .mock_provider import MockSDKProvider

# the video we are testing with has 1990 frames (see test/assets/person_example/img), so the
# iterator should have this many elements
EXPECTED_ITERATOR_LEN = 1990


class MockSDKProviderTests(unittest.TestCase):
    """
    Simple tests to verify the MockSDKProvider is working.

    The MockSDKProvider is a piece of test infrastructure that provides a convienient
    interface for mocking out the SDK/Groundlight service when testing tracking.
    """

    def test_mock_sdk_provider_happy_path_imagequery(self):
        """
        Simple e2e test of MockSDKProvider for return_mode = imagequery
        """
        test_images_path = "test/assets/person_example/img"
        test_detections_path = "test/assets/person_example/detections.jsonl"
        dataset = MockSDKProvider(
            images_path=test_images_path, detections_path=test_detections_path, return_mode="imagequery"
        )

        assert len(dataset) == EXPECTED_ITERATOR_LEN

        # closely examine each element returned to verify the typings
        for example in dataset:
            # each example should have three parts, an image, an imagequery, and a list of metadata dicts per detection
            self.assertEqual(len(example), 3)
            assert isinstance(example[0], Image.Image)
            assert isinstance(example[1], ImageQuery)
            assert isinstance(example[2], list)
            assert all(isinstance(i, dict) for i in example[2])

        # performance check, verifies we can iterate through the data quickly.
        # Separated this from the loop above, as it is doing real work on each example
        start = time.time()
        for _ in dataset:
            pass
        end = time.time()

        total_time = end - start
        time_per_element = total_time / len(dataset)

        assert total_time < 0.2  # noqa: PLR2004
        assert time_per_element < 1e-4  # noqa: PLR2004

    def test_mock_sdk_provider_happy_path_roi(self):
        """
        Simple e2e test of MockSDKProvider for return_mode = roi
        """
        test_images_path = "test/assets/person_example/img"
        test_detections_path = "test/assets/person_example/detections.jsonl"
        dataset = MockSDKProvider(images_path=test_images_path, detections_path=test_detections_path, return_mode="roi")

        assert len(dataset) == EXPECTED_ITERATOR_LEN
        # closely examine each element returned to verify the typings
        for example in dataset:
            # each example should have three parts, an image, a list of ROIs, and a list of metadata dicts per detection
            self.assertEqual(len(example), 3)
            assert isinstance(example[0], Image.Image)
            assert isinstance(example[1], list)
            assert all(isinstance(i, ROI) for i in example[1])
            assert isinstance(example[2], list)
            assert all(isinstance(i, dict) for i in example[2])

        # performance check, verifies we can iterate through the data quickly.
        # Separated this from the loop above, as it is doing real work on each example
        start = time.time()
        for _ in dataset:
            pass
        end = time.time()

        total_time = end - start
        time_per_element = total_time / len(dataset)

        assert total_time < 0.2  # noqa: PLR2004
        assert time_per_element < 1e-4  # noqa: PLR2004

    def test_mock_sdk_provider_metadata(self):
        """
        This test verifies that we properly package metadata on an example that has non-standard fields in the
        detections.
        """
        test_images_path = "test/assets/volleyball_example/img"
        test_detections_path = "test/assets/volleyball_example/detections.jsonl"
        dataset = MockSDKProvider(
            images_path=test_images_path, detections_path=test_detections_path, return_mode="imagequery"
        )

        for _, _, metadata in dataset:
            for detection_metadata in metadata:
                assert "track_id" in detection_metadata, "track_id should be in each detection metadata"
