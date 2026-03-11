from abc import ABC, abstractmethod
from typing import Optional

from groundlight import ROI, ImageQuery

from tracking.track import Track
from tracking.type_definitions import ImageType


class TrackerABC(ABC):
    """
    Abstract base class for all trackers.
    """

    @abstractmethod
    def __init__(self, classes_to_track: Optional[list[str]] = None):
        """
        Initializes the tracker.

        :param classes_to_track: A list of classes to track. If None, all classes will be tracked.
        """

    @abstractmethod
    def update_from_image_query(self, image_query: ImageQuery, frame: ImageType) -> None:
        """
        Parses the image query's result, only keeping self.classes_to_track, and constructing the input for the wrapped
        tracker
        """

    @abstractmethod
    def update_from_rois(self, rois: list[ROI], frame: ImageType) -> None:
        """
        Second interface for updating the tracker. Assumes the ROIs have already been preprocessed,
        allows for application specific preprocessing.
        """

    @abstractmethod
    def reset(self):
        """
        Resets the wrapped tracker's internal state.
        """

    @abstractmethod
    def get_tracks(self) -> list[Track]:
        """
        Gets the tracks from the wrapped tracker and returns a list of the current tracks as Track objects.
        """
