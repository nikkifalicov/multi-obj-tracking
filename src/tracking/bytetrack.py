from typing import Dict, Optional

import supervision as sv
from groundlight import ROI

from tracking.bbox_utils import pixels_to_normalized
from tracking.track import Track
from tracking.tracker_base import TrackerBase
from tracking.type_definitions import ImageType


class ByteTrackTracker(TrackerBase):
    """
    ByteTrack tracker for use with the Groundlight Python SDK. Inherits from TrackerBase
    and extends its functionality.

    Under the hood, we use the ByteTrack implementation from Roboflow's supervision package

    Example usage:
    ```python
    from groundlight import Groundlight
    from tracking.bytetrack import ByteTrack

    gl = Groundlight()
    object_detector = gl.create_counting_detector(
        name="people_counter",
        query="how many people are in the image?",
        class_names=["person"],
        max_count=10,
        confidence_threshold=0.75,
        patience_time=30.0,
    )


    tracker = ByteTrackTracker(
        image_width=1024,
        image_height=1024,
        classes_to_track=["person"]
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
        Initializes the tracker

        :param classes_to_track: A list of classes to track. If None, all classes will be tracked. If not None, the
            tracker will prune detections to only include these classes.
        :param image_width: Width of the input images in pixels.
        :param image_height: Height of the input images in pixels.
        :param kwargs: Additional configuration parameters for the internal ByteTrack tracker.
            See https://supervision.roboflow.com/trackers/#bytetrack for a full list of available parameters.

            At the time of writing, the available parameters are:
            - track_activation_threshold: float (default: 0.25) - Detection confidence threshold for track activation
            - lost_track_buffer: int (default: 30) - Number of frames to buffer when a track is lost
            - minimum_matching_threshold: float (default: 0.8) - Threshold for matching tracks with detections
            - frame_rate: int (default: 30) - The frame rate of the video
            - minimum_consecutive_frames: int (default: 15) - Number of consecutive frames that an object must be
                tracked before it is considered a 'valid' track. Increasing the value prevents the creation of
                accidental tracks from false detection or double detection, but risks missing shorter tracks
        """
        super().__init__(image_width=image_width, image_height=image_height, classes_to_track=classes_to_track)

        # we override the default value of 1 because we found that increasing minimum_consecutive_frames decreases
        # fragmentation
        if "minimum_consecutive_frames" not in kwargs:
            kwargs["minimum_consecutive_frames"] = 15

        self._internal_tracker = sv.ByteTrack(**kwargs)

        # stores the number of frames since the last match for each track id. Used to determine if tracks are still
        # alive
        self._num_frames_since_last_update: Dict[int, Optional[int]] = {}

    def _update_common(self, rois: list[ROI], frame: ImageType) -> None:
        """
        Internal common method that handles the common tracking logic for both update methods.
        """
        rois = self._filter_to_classes_to_track(rois)
        detections = self._rois_to_detections(rois)

        # get the tracks that matched, and also newly initialized tracks
        self._current_tracks = self._internal_tracker.update_with_detections(detections)

        # Update class cache for matched tracks
        if self._current_tracks is not None and len(self._current_tracks) > 0:
            if self._current_tracks.tracker_id is not None and self._current_tracks.class_id is not None:
                for track_id, class_id in zip(self._current_tracks.tracker_id, self._current_tracks.class_id):
                    if track_id != -1:
                        self._track_id_to_class_id[int(track_id)] = int(class_id)
                        # initialize entry, but set it to None to indicate an updated track
                        self._num_frames_since_last_update[int(track_id)] = None

        # update the time we saw each track last
        for track_id, value in list(self._num_frames_since_last_update.items()):
            # if the last frame we saw this track is None, it means it was just seen, else
            # increment the time we last saw the track
            if value is None:
                self._num_frames_since_last_update[track_id] = 0
            else:
                self._num_frames_since_last_update[track_id] += 1  # type: ignore

            # if the track has exceeded its lifespan, remove it
            if self._num_frames_since_last_update[track_id] > self._internal_tracker.max_time_lost:  # type: ignore
                self._num_frames_since_last_update.pop(track_id)

    def reset(self):
        """
        Resets the tracker's internal state.
        """
        super().reset()
        # reset the state of the wrapped tracker
        self._internal_tracker.reset()

        # reset internal state variables unique to implmentation
        self._num_frames_since_last_update = {}

    def _create_track(self, track_id, class_name, bbox_tlwh, frames_since_last_update):
        """
        Helper function to take in information about the track and return a Track object
        representing the relevant information

        Args:
            track_id: int - Unique tracker id for the current track
            class_name: str - Class name for the track
            bbox_tlwh: [int] - Bounding box information for the current track in top, left, width, height format
                expressed in pixels
            frames_since_last_update: int - number of frames since this track was last updated
        Returns:
            Track object representing the track's information
        """
        x1, y1, width, height = bbox_tlwh
        x2 = x1 + width
        y2 = y1 + height

        # clamp coordinates to image bounds
        x1 = max(0, min(x1, self.image_width))
        y1 = max(0, min(y1, self.image_height))
        x2 = max(0, min(x2, self.image_width))
        y2 = max(0, min(y2, self.image_height))

        # ensure x2, y2 are no smaller than x1, y1
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
        track = Track(
            id=int(track_id),
            bbox=normalized_bbox,
            class_name=class_name,
            frames_since_last_update=frames_since_last_update,
        )
        return track

    def _get_alive_tracks(self, track_list: list):
        """
        Helper function that takes in a list of tracks, and returns all tracks that
        matched on the most recent detection, as well as tracks that didn't match on
        the most recent detection but have not been deleted yet.

        Args:
            track_list: list[STrack] - list of supervision STrack tracks
        Returns:
            list of Track objects for all alive tracks, along with a set of all track id's that correspond
            to alive tracks
        """
        tracks: list[Track] = []
        class_id_to_name = {v: k for k, v in self.class_name_to_class_id.items()}
        alive_track_ids = set()
        max_time_lost = self._internal_tracker.max_time_lost

        for track in track_list:
            track_id = track.external_track_id
            if (
                track_id in self._num_frames_since_last_update
                and self._num_frames_since_last_update[track_id] < max_time_lost  # type: ignore
            ):
                # mark the id as alive
                alive_track_ids.add(track_id)

                # obtain information about the track
                class_id = self._track_id_to_class_id[track_id]
                class_name = class_id_to_name[class_id]
                frames_since_last_update = self._num_frames_since_last_update[track_id]
                track_tlwh = track.tlwh.tolist()

                # create Track object
                track_object = self._create_track(track_id, class_name, track_tlwh, frames_since_last_update)

                tracks.append(track_object)

        return tracks, alive_track_ids

    def get_tracks(self, *, max_frames_since_last_update: Optional[int] = None) -> list[Track]:
        """
        Returns tracks that matched with the most recent detections in addition to tracks that are alive but unmatched.

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
                f"max_frames_since_last_update must be None or >= 0, but got {max_frames_since_last_update}"
            )

        # lost tracks are tracks that didn't match during the most recent detection but have not
        # exceeded the lost track buffer. tracked tracks are tracks that matched during the most recent detection
        track_list = self._internal_tracker.tracked_tracks + self._internal_tracker.lost_tracks

        tracks, alive_track_ids = self._get_alive_tracks(track_list)

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

        # clean up cache
        self._remove_dead_tracks_from_cache(alive_track_ids=alive_track_ids)

        return tracks
