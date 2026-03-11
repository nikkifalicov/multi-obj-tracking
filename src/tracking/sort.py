from typing import Optional

from groundlight import ROI
from trackers import SORTTracker as RoboflowSORTTracker
from trackers.utils.sort_utils import (
    get_alive_trackers,
)

from tracking.bbox_utils import pixels_to_normalized
from tracking.track import Track
from tracking.tracker_base import TrackerBase
from tracking.type_definitions import ImageType


class SORTTracker(TrackerBase):
    """
    SORT tracker for use with the Groundlight Python SDK. Inherits from TrackerBase and extends functionality.

    Under the hood, we use the SORT implementation from the trackers package.

    Example usage:
    ```python
    from groundlight import Groundlight
    from tracking.sort import SORTTracker

    gl = Groundlight()
    object_detector = gl.create_counting_detector(
        name="people_counter",
        query="how many people are in the image?",
        class_names=["person"],
        max_count=10,
        confidence_threshold=0.75,
        patience_time=30.0,
    )

    tracker = SORTTracker(
        image_width=1024,
        image_height=1024,
        classes_to_track=["person"],
    )

    images = [img1, img2, img3, ... ] # images coming from some iterable or stream

    for image in images:
        iq = gl.ask_ml(object_detector, image)
        tracker.update_from_image_query(iq, image)

        # use tracks for your application
        tracks = tracker.get_tracks()

    ```
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        classes_to_track: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Initializes the tracker.

        :param classes_to_track: A list of classes to track. If None, all classes will be tracked. If not None, the
            tracker will prune detections to only include these classes.
        :param image_width: Width of the input images in pixels.
        :param image_height: Height of the input images in pixels.
        :param kwargs: Additional configuration parameters for the internal SORT tracker.
            See https://trackers.roboflow.com/develop/trackers/core/sort/tracker/ for a full list of available
            parameters.

            At the time of writing, the available parameters are:
            - lost_track_buffer: int (default: 30) - Number of frames to buffer when a track is lost
            - frame_rate: float (default: 30.0) - Frame rate of the video (frames per second)
            - track_activation_threshold: float (default: 0.25) - Detection confidence threshold for track activation
            - minimum_consecutive_frames: int (default: 3) - Number of consecutive frames that an object must be tracked
                before it is considered valid
            - minimum_iou_threshold: float (default: 0.3) - IOU threshold for associating detections to existing tracks
        """
        super().__init__(image_width=image_width, image_height=image_height, classes_to_track=classes_to_track)
        self._internal_tracker = RoboflowSORTTracker(**kwargs)

    def _update_common(self, rois: list[ROI], frame: ImageType) -> None:  # pylint: disable=unused-argument
        """
        Internal common method that handles the common tracking logic for both update methods.
        Note: frame parameter is ignored since SORT only uses bounding box information but is included for inheritance
        purposes.

        """
        rois = self._filter_to_classes_to_track(rois)
        detections = self._rois_to_detections(rois)

        self._current_tracks = self._internal_tracker.update(
            detections=detections,
        )

        # Update class cache for matched tracks
        if self._current_tracks is not None and len(self._current_tracks) > 0:
            if self._current_tracks.tracker_id is not None and self._current_tracks.class_id is not None:
                for track_id, class_id in zip(self._current_tracks.tracker_id, self._current_tracks.class_id):
                    if track_id != -1:
                        self._track_id_to_class_id[int(track_id)] = class_id

    def reset(self):
        """
        Resets the tracker's internal state.
        """
        super().reset()
        # reset the state of the wrapped tracker
        self._internal_tracker.reset()

    def get_tracks(self, *, max_frames_since_last_update: Optional[int] = None) -> list[Track]:
        """
        Returns tracks that match with the most recent detections in addition to tracks that are alive but unmatched.

        Args:
            max_frames_since_last_update: Optional[int] - Maximum number of frames since the last update for a track to
                be included in the returned list of tracks. If None, all tracks will be included. Must be >= 0. If 0,
                only tracks that have been matched with a detection on the most recent update will be included.
                Otherwise, this function might return tracks that are still alive, but were not matched with a
                detection on the most recent update.
        """
        # validate user input
        if max_frames_since_last_update is not None and max_frames_since_last_update < 0:
            raise ValueError(
                "max_frames_since_last_update must be None or >= 0, but got {max_frames_since_last_update=}"
            )

        # first, collect tracks from the most recent update
        tracks: list[Track] = []

        # first, get tracks that have been matched with a detection on the most recent update
        matched_tracks, matched_track_ids = self._get_matched_tracks()
        tracks.extend(matched_tracks)

        # then, collect tracks that are alive but did not match with any detections during the most recent update
        # alive_track_ids is the set of track ids that are alive (matched or unmatched)
        alive_unmatched_tracks, alive_track_ids = self._get_alive_unmatched_tracks(matched_track_ids=matched_track_ids)
        tracks.extend(alive_unmatched_tracks)

        # Clean up dead tracks from cache that are no longer alive
        self._remove_dead_tracks_from_cache(alive_track_ids=alive_track_ids)

        # filter out tracks that have exceeded the max_frames_since_last_update
        if max_frames_since_last_update is not None:
            for index, track in enumerate(tracks):
                if track.frames_since_last_update is None:
                    raise ValueError(
                        "Track frames_since_last_update is None. This shouldn't happen as we set it to a value for all "
                        "tracks above."
                    )

                if track.frames_since_last_update > max_frames_since_last_update:
                    tracks.pop(index)

        return tracks

    def _get_matched_tracks(self) -> tuple[list[Track], set[int]]:
        """
        Helper function to get_tracks() that returns tracks that have been matched with a detection on the most recent
        update.

        Returns:
            tracks: list[Track] - Tracks that have been matched with a detection on the most recent update.
            matched_track_ids: set[int] - Set of track ids that have been matched with a detection on the most recent
                update.
        """
        tracks: list[Track] = []
        matched_track_ids = set()

        # Create reverse mapping from class_id to class_name
        class_id_to_name = {v: k for k, v in self.class_name_to_class_id.items()}

        # add tracks from the most recent update to the list
        if self._current_tracks is not None and len(self._current_tracks) > 0:
            # Check that required attributes are not None
            if (
                self._current_tracks.tracker_id is None
                or self._current_tracks.class_id is None
                or self._current_tracks.confidence is None
            ):
                raise ValueError(
                    "The current tracks are missing required attributes. "
                    "This shouldn't happen if the tracker is working properly."
                )

            for i in range(len(self._current_tracks)):
                track_id = self._current_tracks.tracker_id[i]
                if track_id == -1:
                    continue

                # Convert pixel coordinates back to normalized coordinates [0, 1]
                x1, y1, x2, y2 = self._current_tracks.xyxy[i]
                normalized_bbox = pixels_to_normalized(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    image_width=self.image_width,
                    image_height=self.image_height,
                )

                class_id = int(self._current_tracks.class_id[i])
                class_name = class_id_to_name[class_id]

                # Create Track object
                track = Track(
                    id=int(track_id),
                    bbox=normalized_bbox,
                    class_name=class_name,
                    # as we've just matched this track, it has 0 frames since the last update
                    frames_since_last_update=0,
                )

                tracks.append(track)
                matched_track_ids.add(int(track_id))

        return tracks, matched_track_ids

    def _get_alive_unmatched_tracks(  # pylint: disable=too-many-locals
        self, matched_track_ids: set[int]
    ) -> tuple[list[Track], set[int]]:
        """
        Helper function to get_tracks() that returns tracks that are alive but did not match with any detections during
        the most recent update.

        Args:
            matched_track_ids: set[int] - Set of track ids that have been matched with a detection on the most recent
                update. This function will filter out tracks that are in this set as they are already handled by
                _get_matched_tracks().

        Returns:
            A tuple containing:
                tracks: list[Track] - Tracks that are alive but did not match with any detections during the most recent
                    update.
                matched_track_ids: set[int] - Set of track ids that have been matched with a detection on the most
                    recent update.
        """

        tracks: list[Track] = []
        class_id_to_name = {v: k for k, v in self.class_name_to_class_id.items()}

        # an iterable of all alive subtrackers
        # by "subtracker", we mean the internal tracker that is used to track a single object by Roboflow's SORT tracker
        alive_sub_trackers = get_alive_trackers(
            self._internal_tracker.trackers,
            self._internal_tracker.minimum_consecutive_frames,
            self._internal_tracker.maximum_frames_without_update,
        )

        # filter to only subtrackers that have a valid track id and are not in the set of matched track ids
        filtered_sub_trackers = []
        for sub_tracker in alive_sub_trackers:
            if sub_tracker.tracker_id != -1 and sub_tracker.tracker_id not in matched_track_ids:
                filtered_sub_trackers.append(sub_tracker)

        # convert each of them to a Track object
        for sub_tracker in filtered_sub_trackers:
            # Convert pixel coordinates back to normalized coordinates [0, 1]
            x1, y1, x2, y2 = sub_tracker.get_state_bbox()

            # Clamp coordinates to image bounds
            x1 = max(0, min(x1, self.image_width))
            y1 = max(0, min(y1, self.image_height))
            x2 = max(0, min(x2, self.image_width))
            y2 = max(0, min(y2, self.image_height))

            # ensure x2, y2 are no smaller than x1, y1, which can happen if the kalman model predicts x1 or y1 moving
            # faster than x2 or y2
            x2 = max(x1, x2)
            y2 = max(y1, y2)

            # Normalize to [0, 1] range
            normalized_bbox = pixels_to_normalized(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                image_width=self.image_width,
                image_height=self.image_height,
            )

            track_id = sub_tracker.tracker_id

            # Get class name from cache - the class id of the most recent matching detection
            # TODO: Consider using an aggregation over the history of class ids that have been matched to a track_id.
            if track_id not in self._track_id_to_class_id:
                raise ValueError(
                    f"Track id {track_id} not found in class_name_to_class_id mapping. This shouldn't happen if the "
                    "tracker is working properly."
                )

            class_name = class_id_to_name[self._track_id_to_class_id[track_id]]

            track = Track(
                id=int(track_id),
                bbox=normalized_bbox,
                class_name=class_name,
                # no match was found for this track, so we use the value stored in the subtracker
                frames_since_last_update=sub_tracker.time_since_update,
            )
            tracks.append(track)
            matched_track_ids.add(int(track_id))

        return tracks, matched_track_ids
