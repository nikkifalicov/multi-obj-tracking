import unittest
from unittest.mock import patch

import cv2
import numpy as np
import supervision as sv
from groundlight import ROI, BBoxGeometry
from parameterized import parameterized

from tracking.bytetrack import ByteTrackTracker
from tracking.evaluation_utils import (
    assert_metrics,
    evaluate_tracking_performance,
    exact,
)
from tracking.track import Track

from .mock_provider import create_mock_sdk_provider


class ByteTrackTrackerTest(unittest.TestCase):
    """
    Tests to verify that the ByteTrackTracker is working
    """

    @parameterized.expand(
        [
            ("imagequery", "update_from_image_query"),
            ("roi", "update_from_rois"),
        ]
    )
    def test_bytetrack_tracker_basic_e2e(self, return_mode, update_method):  # pylint: disable=too-many-locals
        """
        Basic e2e test of the ByteTrack tracker with different input types on the person example with ground truth data
        """
        dataset, image_width, image_height = create_mock_sdk_provider(
            return_mode=return_mode, detections_mode="groundtruth", example_name="person_example"
        )
        tracker = ByteTrackTracker(image_width=image_width, image_height=image_height)

        # keep track of the tracks returned by the tracker after each frame
        all_tracks = {}

        for frame_idx, (frame, data, _) in enumerate(dataset):
            getattr(tracker, update_method)(data, frame)
            tracks = tracker.get_tracks()
            all_tracks[frame_idx] = tracks

        # Evaluate tracking performance using motmetrics
        metrics = evaluate_tracking_performance(
            dataset=dataset,
            predicted_tracks=all_tracks,
            return_mode=return_mode,
            metrics=["mota"],
        )

        # Assert near perfect performance (since this is ground truth data)
        assert_metrics(metrics, {"mota": 0.95}, "ground truth data on person example")

    def test_bytetrack_tracker_on_volleyball_example_groundtruth(
        self, should_save_video: bool = False
    ):  # pylint: disable=too-many-locals
        """
        Test the ByteTrack tracker on the groundtruth data from the volleyball example (multi-object and multi-class)
        """
        dataset, image_width, image_height = create_mock_sdk_provider(
            return_mode="imagequery", detections_mode="groundtruth", example_name="volleyball_example"
        )
        tracker = ByteTrackTracker(
            image_width=image_width, image_height=image_height, frame_rate=30, minimum_consecutive_frames=15
        )

        label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_RIGHT)
        track_box_annotator = sv.BoxAnnotator(thickness=1)
        annotated_images = []
        tracks_by_frame: dict[int, list[Track]] = {}

        for image, data, _ in dataset:
            tracker.update_from_image_query(data, image)
            tracks = tracker.get_tracks()
            tracks_by_frame[len(tracks_by_frame)] = tracks
            detections = tracker.tracks_to_detections(tracks)

            # convert the image to OpenCV format
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # annotate the image with the tracks, if present
            if detections is None or len(detections) == 0:
                # No tracks to annotate
                annotated_image = cv2_image
            else:
                annotated_image = label_annotator.annotate(cv2_image, detections, labels=detections.tracker_id)
                annotated_image = track_box_annotator.annotate(scene=annotated_image, detections=detections)

            annotated_image = np.array(annotated_image)
            annotated_images.append(annotated_image)

        if should_save_video:
            # save the annotated images as a video
            video_path = "test/assets/volleyball_example/bytetrack_tracker_on_groundtruth_demo.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            writer = cv2.VideoWriter(video_path, fourcc, 30, (image_width, image_height))
            try:
                for annotated_image in annotated_images:
                    writer.write(annotated_image)
            finally:
                writer.release()

        # Evaluate tracking performance using motmetrics
        metrics = evaluate_tracking_performance(
            dataset=dataset,
            predicted_tracks=tracks_by_frame,
            return_mode="imagequery",
            metrics=[
                "mota",
                "num_switches",
                "mostly_tracked",
                "partially_tracked",
                "mostly_lost",
            ],
        )

        assert_metrics(
            metrics,
            {
                "mota": 0.90,
                "num_switches": (0, 16),
                "mostly_tracked": exact(4),
                "partially_tracked": exact(0),
                "mostly_lost": exact(0),
            },
            "ground truth data on volleyball example",
        )

    def test_bytetrack_tracker_on_gl_inferences_person_example(
        self, should_save_video: bool = False
    ):  # pylint: disable=too-many-locals
        """
        Test the ByteTrack tracker on the GL inferences on the person example
        This functions almost as a demo of the tracker in action in addition to a unit test.
        """
        dataset, image_width, image_height = create_mock_sdk_provider(
            return_mode="roi", detections_mode="ml_inferences", example_name="person_example"
        )
        tracker = ByteTrackTracker(
            image_width=image_width,
            image_height=image_height,
            # video specific tracker configuration for noisy data
            minimum_matching_threshold=0.9,
            minimum_consecutive_frames=5,
        )

        label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_RIGHT)
        detection_box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2)
        track_box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=1)

        annotated_images = []
        tracks_by_frame: dict[int, list[Track]] = {}

        for frame_idx, (image, data, _) in enumerate(dataset):
            # Convert PIL Image (RGB) to OpenCV format (BGR)
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            tracker.update_from_rois(data, image)
            tracks_by_frame[frame_idx] = tracker.get_tracks()
            # if frame_idx == 500:
            #     breakpoint()
            for track in tracks_by_frame[frame_idx]:
                assert track.confidence is None, (
                    f"Confidence should be None for track {track.id} at frame {frame_idx} as the ByteTrack tracker"
                    "does not provide confidence scores for tracks"
                )

            # annotate the image with the raw detections, if present
            raw_detections = tracker._rois_to_detections(data)
            annotated_image = detection_box_annotator.annotate(image_cv, raw_detections)

            if len(data) == 0:
                # write no detections to the annotated image in the bottom right corner in white
                cv2.putText(annotated_image, "No detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # annotate the image with the tracks, if present
            if (
                tracker._current_tracks is None
                or len(tracker._current_tracks) == 0
                or tracker._current_tracks.tracker_id == np.array([-1])
            ):
                # No tracks to annotate
                pass
            else:
                annotated_image = label_annotator.annotate(
                    annotated_image, tracker._current_tracks, labels=tracker._current_tracks.tracker_id
                )
                annotated_image = track_box_annotator.annotate(
                    scene=annotated_image, detections=tracker._current_tracks
                )

            annotated_image = np.array(annotated_image)
            annotated_images.append(annotated_image)

        if should_save_video:
            # save the annotated images as a video
            video_path = "test/assets/person_example/bytetrack_tracker_on_gl_inferences_demo.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            writer = cv2.VideoWriter(video_path, fourcc, 30, (image_width, image_height))
            try:
                for annotated_image in annotated_images:
                    writer.write(annotated_image)
            finally:
                writer.release()

        # Evaluate tracking performance using motmetrics against ground truth
        ground_truth_dataset, _, _ = create_mock_sdk_provider(
            return_mode="roi", detections_mode="groundtruth", example_name="person_example"
        )

        metrics = evaluate_tracking_performance(
            dataset=ground_truth_dataset,
            predicted_tracks=tracks_by_frame,
            return_mode="roi",
            metrics=["mota"],
        )

        # Assert reasonable performance for noisy ML inferences (lower thresholds than ground truth)
        assert_metrics(metrics, {"mota": 0.85}, "GL inferences on person example")

    def test_bytetrack_tracker_on_gl_inferences_volleyball_example(
        self, should_save_video: bool = False
    ):  # pylint: disable=too-many-locals
        """
        Test the ByteTrack tracker on the GL inferences on the volleyball example
        """
        dataset, image_width, image_height = create_mock_sdk_provider(
            return_mode="roi", detections_mode="ml_inferences", example_name="volleyball_example"
        )

        tracker = ByteTrackTracker(
            image_width=image_width,
            image_height=image_height,
            frame_rate=30,
            minimum_consecutive_frames=10,
            # minimum_matching_threshold=0.5,
        )

        label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_RIGHT)
        track_box_annotator = sv.BoxAnnotator(thickness=1)
        annotated_images = []
        tracks_by_frame: dict[int, list[Track]] = {}

        for image, data, _ in dataset:
            tracker.update_from_rois(data, image)
            tracks = tracker.get_tracks()
            tracks_by_frame[len(tracks_by_frame)] = tracks
            detections = tracker.tracks_to_detections(tracks)

            # convert the image to OpenCV format
            cv2_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # annotate the image with the tracks, if present
            if detections is None or len(detections) == 0:
                # No tracks to annotate
                annotated_image = cv2_image
            else:
                annotated_image = label_annotator.annotate(cv2_image, detections, labels=detections.tracker_id)
                annotated_image = track_box_annotator.annotate(scene=annotated_image, detections=detections)

            annotated_image = np.array(annotated_image)
            annotated_images.append(annotated_image)

        if should_save_video:
            # save the annotated images as a video
            video_path = "test/assets/volleyball_example/bytetrack_tracker_on_gl_inferences_demo.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
            writer = cv2.VideoWriter(video_path, fourcc, 30, (image_width, image_height))
            try:
                for annotated_image in annotated_images:
                    writer.write(annotated_image)
            finally:
                writer.release()

        # Evaluate tracking performance using motmetrics against ground truth
        ground_truth_dataset, _, _ = create_mock_sdk_provider(
            return_mode="roi", detections_mode="groundtruth", example_name="volleyball_example"
        )

        metrics = evaluate_tracking_performance(
            dataset=ground_truth_dataset,
            predicted_tracks=tracks_by_frame,
            return_mode="roi",
            metrics=["mota", "num_switches", "mostly_tracked", "partially_tracked", "mostly_lost"],
        )

        # Assert reasonable performance for noisy ML inferences on volleyball (more lenient than ground truth)
        assert_metrics(
            metrics,
            {
                "mota": 0.85,
                "num_switches": (0, 30),
                "mostly_tracked": (3, 4),
                "partially_tracked": (0, 1),
                "mostly_lost": exact(0),
            },
            "GL inferences on volleyball example",
        )

    def test_reset(self):
        """
        Verifies that the ByteTrack tracker's reset method correctly resets all internal state,
        including calling reset on the wrapped tracker.
        """

        tracker = ByteTrackTracker(image_width=100, image_height=100)

        # Add some state to the tracker by creating class IDs
        tracker._get_class_id("person")
        tracker._get_class_id("car")
        tracker._get_class_id("bicycle")

        # Verify state exists before reset
        assert len(tracker.class_name_to_class_id) == 3  # noqa: PLR2004
        assert tracker.class_name_to_class_id == {"person": 0, "car": 1, "bicycle": 2}

        # Test that tracker.reset() clears the _current_tracks list and _track_id_to_class_id cache
        # Create ROIs to send to the tracker
        test_rois = [
            ROI(
                label="person",
                score=0.9,
                geometry=BBoxGeometry(left=0.1, top=0.2, right=0.5, bottom=0.6, x=0.3, y=0.4),
            ),
            ROI(
                label="car",
                score=0.8,
                geometry=BBoxGeometry(left=0.6, top=0.3, right=0.9, bottom=0.7, x=0.75, y=0.5),
            ),
        ]

        # Send the same ROIs multiple times to establish tracking
        # The default minimum_consecutive_frames is 3, so we send 4 times to be safe
        frame_mock = np.zeros((100, 100, 3))
        for _ in range(16):
            tracker.update_from_rois(test_rois, frame=frame_mock)

        # Verify that tracking is established (_current_tracks has data)
        assert tracker._current_tracks is not None, "Expected _current_tracks to be set after multiple updates"
        assert len(tracker._current_tracks) > 0, "Expected at least one detection in _current_tracks"

        # Verify that the track ID to class ID cache is populated
        # The cache should have entries for tracks that were matched
        assert len(tracker._track_id_to_class_id) > 0, "Expected _track_id_to_class_id cache to be populated"

        # Verify cache contains valid mappings (track_id -> class_id)
        for track_id, class_id in tracker._track_id_to_class_id.items():
            assert isinstance(track_id, int), f"Expected track_id to be int, got {type(track_id)}"
            assert isinstance(class_id, (int, np.integer)), f"Expected class_id to be int, got {type(class_id)}"
            assert track_id >= 0, f"Expected track_id >= 0, got {track_id}"
            assert class_id >= 0, f"Expected class_id >= 0, got {class_id}"

        # Mock the internal tracker's reset method to verify it's called on reset
        with patch.object(tracker._internal_tracker, "reset") as mock_reset:
            # Call reset
            tracker.reset()

            # Verify the wrapped tracker's reset was called exactly once
            mock_reset.assert_called_once()

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

    def test_track_id_to_class_id_cache_cleanup(self):
        """
        Test that dead tracks are removed from the _track_id_to_class_id cache while alive tracks remain.
        """
        # Use a shorter buffer to make tracks die faster for testing
        tracker = ByteTrackTracker(
            image_width=100,
            image_height=100,
            lost_track_buffer=5,  # Tracks die after 5 frames without update
            minimum_consecutive_frames=1,  # Tracks become valid after 1 frame
        )

        # Create initial ROIs for two different objects
        person_roi = ROI(
            label="person",
            score=0.9,
            geometry=BBoxGeometry(left=0.1, top=0.2, right=0.4, bottom=0.6, x=0.25, y=0.4),
        )
        car_roi = ROI(
            label="car",
            score=0.8,
            geometry=BBoxGeometry(left=0.6, top=0.3, right=0.9, bottom=0.7, x=0.75, y=0.5),
        )

        # Establish two tracks by sending both ROIs together initially
        initial_rois = [person_roi, car_roi]
        frame_mock = np.zeros((100, 100, 3))
        for _ in range(2):  # Send a few times to establish tracking
            tracker.update_from_rois(initial_rois, frame=frame_mock)

        # Get tracks and verify both are being tracked
        tracks = tracker.get_tracks()
        assert len(tracks) == len(initial_rois), "Expected 2 tracks to be established"

        # Verify cache is populated for both tracks
        assert len(tracker._track_id_to_class_id) == len(
            initial_rois
        ), f"Expected at least 2 entries in cache, got {len(tracker._track_id_to_class_id)}"

        # Now stop providing detections for the "person" track, continue with "car"
        # This should cause the person track to eventually die
        car_only_rois = [car_roi]

        # Send updates with only the car ROI for enough frames to kill the person track
        # We need to exceed the lost_track_buffer (5 frames)
        frame_mock = np.zeros((100, 100, 3))
        for _ in range(7):
            tracker.update_from_rois(car_only_rois, frame=frame_mock)

            # We must call get_tracks to trigger cache cleanup
            tracks = tracker.get_tracks()

        # Verify that the cache has been cleaned up
        final_cache_size = len(tracker._track_id_to_class_id)
        assert final_cache_size == len(car_only_rois), (
            f"Expected cache to be cleaned up. Initial: 2, Final: {final_cache_size}. "
            "Only one track should remain in the cache."
        )

        # Verify that we still have at least one active track (the car)
        final_tracks = tracker.get_tracks()
        assert len(final_tracks) >= 1, "Expected at least one active track (car) to remain"

        # Verify that the remaining tracks in the cache correspond to active tracks
        active_track_ids = {track.id for track in final_tracks}
        cached_track_ids = set(tracker._track_id_to_class_id.keys())

        assert (
            cached_track_ids == active_track_ids
        ), f"Cache should only contain active tracks. Active: {active_track_ids}, Cached: {cached_track_ids}"

    def test_predicted_tracks_use_cached_class_names_and_frames_since_last_update_increments(self):
        """
        Test that unmatched tracks continue to be returned with the cached class names and frames_since_last_update
        increments properly.
        """
        # Use shorter buffer to control when tracks die
        tracker = ByteTrackTracker(
            image_width=100,
            image_height=100,
            lost_track_buffer=5,  # Tracks die after 5 frames without update
            minimum_consecutive_frames=1,  # Tracks become valid after 1 frame
        )

        # Create ROI for a person
        person_roi = ROI(
            label="person",
            score=0.9,
            geometry=BBoxGeometry(left=0.1, top=0.2, right=0.4, bottom=0.6, x=0.25, y=0.4),
        )

        # Establish track by sending ROI multiple times
        frame_mock = np.zeros((100, 100, 3))
        for _ in range(2):
            tracker.update_from_rois([person_roi], frame=frame_mock)

        # Get track and verify it's matched
        matched_tracks = tracker.get_tracks()
        assert len(matched_tracks) == 1, "Expected exactly 1 track to be established"
        track = matched_tracks[0]
        assert track.frames_since_last_update == 0, "Newly matched track should have frames_since_last_update=0"
        assert track.class_name == "person", "Track should have person class name"
        assert track.confidence is None, "Confidence should be None for all tracks from ByteTrack"

        # Store track info for comparison
        track_id = track.id
        expected_class = track.class_name

        # Verify cache is populated
        assert track_id in tracker._track_id_to_class_id, "Track should be in cache"

        # Stop providing detections - track should become predicted but remain alive
        empty_rois = []

        # Test frames 1-4: track should be alive but unmatched
        frame_mock = np.zeros((100, 100, 3))
        for frame in range(1, 5):  # frames 1, 2, 3, 4
            tracker.update_from_rois(empty_rois, frame=frame_mock)
            tracks = tracker.get_tracks()

            # Track should still be alive
            assert len(tracks) == 1, f"Expected track to be alive at frame {frame}"
            predicted_track = tracks[0]

            # Verify frames_since_last_update increments
            assert (
                predicted_track.frames_since_last_update == frame
            ), f"Expected frames_since_last_update={frame}, got {predicted_track.frames_since_last_update}"

            # Verify class name comes from cache
            assert (
                predicted_track.class_name == expected_class
            ), f"Expected cached class name '{expected_class}', got '{predicted_track.class_name}'"

            # Verify that confidence remains None for predicted tracks
            assert predicted_track.confidence is None, "Confidence should be None for predicted tracks"

            # Verify it's the same track ID
            assert predicted_track.id == track_id, "Should be same track ID"
        # Frame 5: track should die (exceeds lost_track_buffer)
        frame_mock = np.zeros((100, 100, 3))
        tracker.update_from_rois(empty_rois, frame=frame_mock)
        tracks = tracker.get_tracks()
        assert len(tracks) == 0, "Track should be dead at frame 5"

        # Verify cache is cleaned up
        assert track_id not in tracker._track_id_to_class_id, "Dead track should be removed from cache"

    def test_track_class_changes_over_time(self):
        """
        Test that cache uses most recent class when a track changes class over time.
        """
        # Use longer buffer to keep tracks alive through class changes
        tracker = ByteTrackTracker(
            image_width=100,
            image_height=100,
            lost_track_buffer=10,  # Keep tracks alive longer
            minimum_consecutive_frames=1,  # Tracks become valid after 1 frame
        )

        # Create ROI for initial "person" detection at a specific location
        person_roi = ROI(
            label="person",
            score=0.9,
            geometry=BBoxGeometry(left=0.1, top=0.2, right=0.4, bottom=0.6, x=0.25, y=0.4),
        )

        # Establish track with "person" class
        frame_mock = np.zeros((100, 100, 3))
        for _ in range(2):
            tracker.update_from_rois([person_roi], frame=frame_mock)

        # Verify initial track and cache state
        tracks = tracker.get_tracks()
        assert len(tracks) == 1, "Expected exactly 1 track to be established"
        initial_track = tracks[0]
        assert initial_track.class_name == "person", "Track should initially have person class"
        assert initial_track.confidence is None, "Confidence should be None for initial track"

        track_id = initial_track.id
        person_class_id = tracker.class_name_to_class_id["person"]
        assert tracker._track_id_to_class_id[track_id] == person_class_id, "Cache should have person class"

        # Let track become predicted for a few frames
        frame_mock = np.zeros((100, 100, 3))
        for _ in range(3):
            tracker.update_from_rois([], frame=frame_mock)
            tracks = tracker.get_tracks()
            assert len(tracks) == 1, "Track should remain alive"
            assert tracks[0].class_name == "person", "Predicted track should keep cached person class"

        # Now provide a "car" detection at the same location
        # The tracker should associate this with the existing track and update the class
        car_roi = ROI(
            label="car",
            score=0.8,
            geometry=BBoxGeometry(left=0.1, top=0.2, right=0.4, bottom=0.6, x=0.25, y=0.4),  # Same location
        )

        # Send car detection - should match with existing track and update class
        tracker.update_from_rois([car_roi], frame=frame_mock)
        tracks = tracker.get_tracks()

        # Verify track updated to new class
        assert len(tracks) == 1, "Should still have exactly 1 track"
        updated_track = tracks[0]
        assert updated_track.id == track_id, "Should be the same track ID"
        assert updated_track.class_name == "car", "Track should now have car class"
        assert updated_track.frames_since_last_update == 0, "Track should be freshly matched"
        assert updated_track.confidence is None, "Confidence should be None after class update"

        # Verify cache was updated with new class
        car_class_id = tracker.class_name_to_class_id["car"]
        assert tracker._track_id_to_class_id[track_id] == car_class_id, "Cache should now have car class"

    def test_max_frames_since_last_update_validation(self):
        """
        Test that max_frames_since_last_update parameter validation raises ValueError for negative values.
        """
        tracker = ByteTrackTracker(image_width=100, image_height=100)

        # Valid values should not raise
        tracker.get_tracks(max_frames_since_last_update=None)
        tracker.get_tracks(max_frames_since_last_update=0)
        tracker.get_tracks(max_frames_since_last_update=5)

        # Invalid values should raise ValueError
        with self.assertRaises(ValueError):
            tracker.get_tracks(max_frames_since_last_update=-1)

    def test_max_frames_since_last_update_filtering(self):
        """
        Test that max_frames_since_last_update parameter filters tracks by frames_since_last_update.
        """
        tracker = ByteTrackTracker(image_width=100, image_height=100, lost_track_buffer=5, minimum_consecutive_frames=1)

        # Create a track
        roi = ROI(
            label="person", score=0.9, geometry=BBoxGeometry(left=0.1, top=0.2, right=0.4, bottom=0.6, x=0.25, y=0.4)
        )
        frame_mock = np.zeros((100, 100, 3))
        for _ in range(2):
            tracker.update_from_rois([roi], frame=frame_mock)
        # check that the track was initialized with the correct frames since last update

        # Make track alive but unmatched (frames_since_last_update=1)
        tracker.update_from_rois([], frame=frame_mock)
        tracks = tracker.get_tracks()
        assert len(tracks) == 1
        assert tracks[0].frames_since_last_update == 1

        # Test filtering
        assert len(tracker.get_tracks(max_frames_since_last_update=0)) == 0  # No matched tracks
        assert len(tracker.get_tracks(max_frames_since_last_update=1)) == 1  # Include predicted track
        assert len(tracker.get_tracks(max_frames_since_last_update=None)) == 1  # Include all tracks
