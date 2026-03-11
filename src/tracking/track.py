from dataclasses import dataclass
from typing import Optional


@dataclass
class Track:
    """
    A track represents a single object that is being tracked.

    Attributes:
        id: int - unique identifier for the track
        bbox: tuple[float, float, float, float] - normalized bounding box coordinates (x1, y1, x2, y2) from [0, 1]
        class_name: str - name of the class being tracked
        confidence: Optional[float] - confidence score for the track - from [0, 1] or None if not available.
        number_of_successful_updates: Optional[int] - number of successful updates for the track - also refered to as
        "hits" in tracking nomenclature. None if not available.
        frames_since_last_update: Optional[int] - number of frames since the last update for the track. None if not
        available.
        time_since_update_seconds: Optional[float] - time since the last update for the track. None if not available.
    """

    id: int
    bbox: list[float]
    class_name: str
    confidence: Optional[float] = None
    number_of_successful_updates: Optional[int] = None
    frames_since_last_update: Optional[int] = None
    time_since_update_seconds: Optional[float] = None

    def __post_init__(self):
        """
        This function is called after a Track is initialized.
        """
        self._validate_data()

    def _validate_data(self):
        """
        This is a no-op if the object instance is valid. Otherwise, it will raise a ValueError.
        """

        # validate id - it should be an integer >= 0
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError(f"Track id must be an integer >= 0, but got {self.id=}")

        # validate bbox - it should be a tuple of 4 floats in [0, 1], x2 >= x1, and y2 >= y1
        if (
            not isinstance(self.bbox, list)
            or len(self.bbox) != 4  # noqa: PLR2004
            or any(not isinstance(coord, float) or coord < 0 or coord > 1 for coord in self.bbox)
        ):
            raise ValueError(f"Track bbox must be a list of 4 floats in [0, 1], but got {self.bbox=}")
        x1, y1, x2, y2 = self.bbox
        if x2 < x1 or y2 < y1:  # noqa: PLR2004
            raise ValueError(f"Track bbox must have x2 >= x1 and y2 >= y1, but got {self.bbox=}")

        # validate class_name - it should be a non-empty string
        if not isinstance(self.class_name, str) or len(self.class_name) == 0:
            raise ValueError(f"Track class_name must be a non-empty string, but got {self.class_name=}")

        # validate confidence - it should be a float in [0, 1] or None
        if self.confidence is not None and (
            not isinstance(self.confidence, float) or self.confidence < 0 or self.confidence > 1
        ):
            raise ValueError(f"Track confidence must be a float in [0, 1] or None, but got {self.confidence=}")

        # validate number_of_successful_updates - it should be an integer >= 0 or None
        if self.number_of_successful_updates is not None and (
            not isinstance(self.number_of_successful_updates, int) or self.number_of_successful_updates < 0
        ):
            raise ValueError(
                "Track number_of_successful_updates must be an integer >= 0 or None, "
                f"but got {self.number_of_successful_updates=}"
            )

        # validate frames_since_last_update - it should be an integer >= 0 or None
        if self.frames_since_last_update is not None and (
            not isinstance(self.frames_since_last_update, int) or self.frames_since_last_update < 0
        ):
            raise ValueError(
                "Track frames_since_last_update must be an integer >= 0 or None, "
                f"but got {self.frames_since_last_update=}"
            )

        # validate time_since_update_seconds - it should be a float >= 0 or None
        if self.time_since_update_seconds is not None and (
            not isinstance(self.time_since_update_seconds, float) or self.time_since_update_seconds < 0
        ):
            raise ValueError(
                "Track time_since_update_seconds must be a float >= 0 or None, "
                f"but got {self.time_since_update_seconds=}"
            )
