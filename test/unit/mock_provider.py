import glob
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterator, Union

import cv2
from groundlight import ImageQueryTypeEnum, ResultTypeEnum
from groundlight.client import ROI, BBoxGeometry, ImageQuery
from PIL import Image

from tracking.type_definitions import ImageType


class MockSDKProvider:
    """
    This class creates an iterable of mock Groundlight SDK objects for testing trackers
    """

    def __init__(
        self,
        images_path: str,
        detections_path: str,
        return_mode: str,
    ):
        """
        This class provides an iterator over a video you want to track. You must provide paths to a folder containing
        the frames in the video and to a file with the detections in each frame. The iterator returns the data in frame
        order and packaged for you to pass to the relevant Tracker.update* method. This allows us to create datasets
        of ground truth boxes or boxes derived from inference for testing tracking algorithms without requiring
        calls to the SDK or Groundlight service at test time.

        Arguments:
        images_path: str - path to folder containing image frames to use
        detections_path: str - path to a jsonl file mapping frame indicies to detections
            Each row in jsonl should be a dictionary representing a single detection with the following schema:
            {
                frame_index: int
                class_name: str
                box: list[int] - [x1, y1, x2, y2] - 0 to 1 normalized
                confidence: float - between 0 and 1
                # any additional keys will be captured as metadata and returned if requested

            }

        return_mode: str - either "imagequery" or "roi".
            Determines the type of SDK response object the iterator will return.
            If "imagequery", the iterator will retun a tuple: tuple[ImageType, ImageQuery]
            If "roi", the iterator will return a tuple: tuple[ImageType, list[ROI]]
        """

        # scrape images_path for all images in the folder and sort them for iteration
        # store in self.image_paths
        extensions = ["*.png", "*.jpg", "*.jpeg"]
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_path, ext)))
        self.image_paths.sort()

        # load detections from detections_path
        # store in mapping from frame_index -> list[dict]
        self.detections: dict[int, list[dict]] = {}
        with open(detections_path, encoding="utf-8") as file:
            for row in file:
                row_stripped = row.strip()
                detection = json.loads(row_stripped)
                if detection["frame_index"] not in self.detections:
                    self.detections[detection["frame_index"]] = []
                self.detections[detection["frame_index"]].append(detection)

        valid_return_modes = ["imagequery", "roi"]
        if return_mode not in valid_return_modes:
            raise ValueError(f"return_mode not valid. Valid return modes={valid_return_modes}")
        self.return_mode = return_mode

        # keeps track of which frame index to return next
        self._index = 0

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> Iterator:
        # reset iterator
        self._index = 0
        return self

    def __next__(self) -> tuple[ImageType, Union[ImageQuery, list[ROI]], list[dict]]:
        if self._index >= len(self.image_paths):
            raise StopIteration
        image_path: str = self.image_paths[self._index]
        detections: Union[list[dict], None] = self.detections.get(self._index, None)

        inference_data: ImageQuery = self._package_detections(detections=detections)
        if self.return_mode == "roi":
            inference_data = inference_data.rois

        metadata = [self._package_metadata(detection) for detection in detections] if detections else []

        image: Image.Image = Image.open(image_path)

        self._index += 1

        return (image, inference_data, metadata)

    def _package_detections(self, detections: Union[list[dict], None]) -> ImageQuery:
        rois = []

        if detections is not None:
            for detection in detections:
                label = detection["class_name"]
                score = detection["confidence"]
                box = detection["box"]

                left = box[0]
                top = box[1]
                right = box[2]
                bottom = box[3]
                x = (left + right) / 2
                y = (top + bottom) / 2

                roi = ROI(
                    label=label,
                    score=score,
                    geometry=BBoxGeometry(left=left, top=top, right=right, bottom=bottom, x=x, y=y),
                )

                rois.append(roi)

        iq = ImageQuery(
            id="iq_mock_iq_id_from_mock_sdk_provider",
            metadata={},
            type=ImageQueryTypeEnum.image_query,
            created_at=datetime.now(),
            query="Mock query from MockSDKProvider",
            detector_id="det_mock_detector_id_from_mock_sdk_provider",
            result_type=ResultTypeEnum.counting,
            # TODO - Do we intend on using this field downstream. If we do, we can mock this field properly
            result=None,
            patience_time=30.0,
            confidence_threshold=75.0,
            rois=rois,
            done_processing=True,
            text=None,
        )

        return iq

    def _package_metadata(self, detection: dict) -> dict:
        """
        Captures all non-standard fields in the detection dictionary and returns them as a dictionary.
        """

        metadata = {}
        for key, value in detection.items():
            if key not in ["frame_index", "class_name", "box", "confidence"]:
                metadata[key] = value
        return metadata


def create_mock_sdk_provider(
    return_mode: str,
    detections_mode: str,
    example_name: str,
) -> tuple[MockSDKProvider, int, int]:
    """
    Creates a MockSDKProvider dataset for testing.
    Returns the dataset, image width, and image height. The image width and height are used to initialize a
    tracker for this dataset.

    return_mode: str - either "imagequery" or "roi" - determines the type of SDK response object the iterator will
        return.
    detections_mode: str - either "groundtruth" or "ml_inferences" - determines the source of the detections,
        either from the GT or from the ML inferences.
    """

    example_path = Path(f"test/assets/{example_name}")
    test_images_path = example_path / "img"

    if not example_path.exists():
        raise ValueError(f"Example directory does not exist: {example_path}")

    if not test_images_path.exists():
        raise ValueError(f"Images directory does not exist: {test_images_path}")

    if detections_mode == "groundtruth":
        test_detections_path = example_path / "detections.jsonl"
    elif detections_mode == "ml_inferences":
        test_detections_path = example_path / "ml_inferences.jsonl"
    else:
        raise ValueError(f"Invalid detections mode: {detections_mode}")

    if not test_detections_path.exists():
        raise ValueError(f"Detections file does not exist: {test_detections_path}")

    dataset = MockSDKProvider(
        images_path=str(test_images_path), detections_path=str(test_detections_path), return_mode=return_mode
    )

    first_image_path = test_images_path / "00000001.jpg"
    if not first_image_path.exists():
        raise ValueError(f"First image file does not exist: {first_image_path}")

    sample_image = cv2.imread(str(first_image_path))
    image_height, image_width = sample_image.shape[:2]

    return dataset, image_width, image_height
