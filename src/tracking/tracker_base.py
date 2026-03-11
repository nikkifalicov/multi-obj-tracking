"""
This file defines logic that is the same across all of the tracking methodologies, but
does not function as a tracker class in itself. Used to reduce duplicated code.
"""

from typing import Optional

import numpy as np
import supervision as sv
from groundlight import ROI, ImageQuery

from tracking.bbox_utils import normalized_to_pixels
from tracking.track import Track
from tracking.tracker_interface import TrackerABC
from tracking.type_definitions import ImageType


class TrackerBase(TrackerABC):
    """
    Non-functional tracker implementation. Does not include any logic for tracking; solely to be used for inheritance
    purposes by other trackers
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        classes_to_track: Optional[list[str]] = None,
    ):
        """
        Initializes variables related to the tracker. Does not actually initialize the tracker.

        :param classes_to_track: A list of classes to track.
        :param image_width: Width of the input images in pixels.
        :param image_height: Height of the input images in pixels.
        :param kwargs: Additional configuration parameters for the internal tracker
        """
        self.classes_to_track = classes_to_track
        self.image_width = image_width
        self.image_height = image_height
        if self.image_width is None or self.image_height is None:
            raise ValueError(
                "Image width and height must be provided during initialization to convert normalized ROI coordinates"
            )

        # This dictionary maps class names to class ids. It updates as we encounter new classes at runtime
        self.class_name_to_class_id: dict[str, int] = {}

        # Placeholder variable; for the detections that were matched to tracks on the most recent call to update
        self._current_tracks: Optional[sv.Detections] = None

        # Placeholder; cache to store track_id -> class_id mapping for tracks that are alive
        self._track_id_to_class_id: dict[int, int] = {}

    def update_from_image_query(self, image_query: ImageQuery, frame: ImageType) -> None:
        """
        Parses the image query's result, only keeping self.classes_to_track.

        Makes a call to the update function, but does nothing unless _update_common is overridden with correct logic.
        If overridden, this function will update the tracker's internal state.
        """
        rois = image_query.rois if image_query.rois else []
        self._update_common(rois, frame)

    def update_from_rois(self, rois: list[ROI], frame: ImageType) -> None:
        """
        Takes in the provided ROIs and makes a call to the update function, but does nothing unless _update_common is
        overridden with correct logic.

        When _update_common is implemented, this function allows for application specific preprocessing of the ROIs
        prior to passing them to the tracker, and updates the tracker based on the provided ROIs
        """
        self._update_common(rois, frame)

    def _update_common(self, rois: list[ROI], frame: ImageType) -> None:  # pylint: disable=unused-argument
        """
        Raises a NotImplementedError when called, unless overridden. When overridden correctly, it handles the common
        tracking logic for both update methods.
        """
        # return an error if this method is not overriden and a user tries to use it
        raise NotImplementedError("Implement logic for _update_common")

    def reset(self):
        """
        Resets the TrackerBase internal state
        """
        # reset the state that we maintain
        self.class_name_to_class_id = {}
        self._current_tracks = None
        self._track_id_to_class_id = {}

    def get_tracks(self, *, max_frames_since_last_update: Optional[int] = None) -> list[Track]:
        """
        Raises NotImplementedError when called. When overridden correctly, returns tracks that match with the most
        recent detections in addition to tracks that are alive but unmatched.

        Args:
            max_frames_since_last_update: Optional[int] - Maximum number of frames since the last update for a track to
                be included in the returned list of tracks.
        """
        raise NotImplementedError("Implement logic for get_tracks")

    def _remove_dead_tracks_from_cache(self, alive_track_ids: set[int]) -> None:
        """
        Removes dead tracks from the cache.

        Args:
            alive_track_ids: set[int] - Set of track ids that are alive (matched or unmatched). This function will
                remove tracks that are not in this set from the cache.
        """
        dead_track_ids = set(self._track_id_to_class_id.keys()) - alive_track_ids
        for dead_track_id in dead_track_ids:
            del self._track_id_to_class_id[dead_track_id]

    def _get_class_id(self, class_name: str) -> int:
        """
        Returns the class id for a given class name.
        If the class name is not in the class_name_to_class_id dictionary, it is added and assigned a new id.
        """
        if class_name not in self.class_name_to_class_id:
            self.class_name_to_class_id[class_name] = len(self.class_name_to_class_id)
        return self.class_name_to_class_id[class_name]

    def _rois_to_detections(self, rois: list[ROI]) -> sv.Detections:
        """
        Convert Groundlight ROIs to supervision library Detections format.

        ROIs use normalized coordinates (0-1), which are converted to pixel coordinates
        using the image dimensions provided during initialization.
        """
        if not rois:
            return sv.Detections.empty()

        xyxy = []
        confidences = []
        class_ids = []

        for roi in rois:
            # Convert normalized coordinates (0-1) to pixel coordinates
            pixel_bbox = normalized_to_pixels(
                x1=roi.geometry.left,
                y1=roi.geometry.top,
                x2=roi.geometry.right,
                y2=roi.geometry.bottom,
                image_width=self.image_width,
                image_height=self.image_height,
            )

            xyxy.append(pixel_bbox)
            confidences.append(roi.score)
            class_ids.append(self._get_class_id(roi.label))

        return sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
        )

    def _filter_to_classes_to_track(self, rois: list[ROI]) -> list[ROI]:
        """
        Filter ROIs to only include specified classes in self.classes_to_track. If self.classes_to_track is None,
        this is a no-op.
        """

        if self.classes_to_track is None:
            return rois

        return [roi for roi in rois if roi.label in self.classes_to_track]

    def tracks_to_detections(self, tracks: list[Track]) -> sv.Detections:  # pylint: disable=too-many-locals
        """
        Convert a list of Track objects to a supervision Detections object.

        This method takes the output from get_tracks() and converts it into the standard
        sv.Detections format, which can be used with supervision's annotation and analysis tools.

        Args:
            tracks: List of Track objects to convert

        Returns:
            sv.Detections: A Detections object containing the track information

        """
        if not tracks:
            return sv.Detections.empty()

        # Convert tracks to arrays
        xyxy = []
        confidences = []
        class_ids = []
        tracker_ids = []
        class_names = []

        for track in tracks:
            # Convert normalized coordinates back to pixel coordinates
            x1, y1, x2, y2 = track.bbox
            pixel_bbox = normalized_to_pixels(
                x1=x1, y1=y1, x2=x2, y2=y2, image_width=self.image_width, image_height=self.image_height
            )

            xyxy.append(pixel_bbox)
            confidences.append(track.confidence)
            tracker_ids.append(track.id)
            class_names.append(track.class_name)

            # Get class_id from class_name using existing mapping
            class_id = self.class_name_to_class_id.get(track.class_name)
            if class_id is None:
                raise ValueError(
                    f"Class name '{track.class_name}' not found in class_name_to_class_id mapping. "
                    "This shouldn't happen if the track was created by this tracker."
                )
            class_ids.append(class_id)

        return sv.Detections(
            xyxy=np.array(xyxy, dtype=np.float32),
            confidence=np.array(confidences, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
            tracker_id=np.array(tracker_ids, dtype=int),
        )
