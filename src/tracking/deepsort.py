from typing import Optional, Union

import numpy as np
import supervision as sv
import torch
from groundlight import ROI
from PIL import Image
from trackers import DeepSORTTracker as RoboflowDeepSORTTracker  # type: ignore[possibly-unbound-import]
from trackers import ReIDModel  # type: ignore[possibly-unbound-import]

from tracking.sort import SORTTracker
from tracking.type_definitions import ImageType


class OptimizedReIDModel(ReIDModel):
    """
    Optimized ReID model that automatically chooses between serial and batched inference
    based on device capabilities.
    """

    def __init__(self, *args, **kwargs):
        """
        args and kwargs are passed to the ReIDModel constructor.
        """
        super().__init__(*args, **kwargs)
        self.can_batch = "cuda" in self.device.type and torch.cuda.is_available()

        self.compiled = False

    @classmethod
    def from_timm(
        cls,
        model_name_or_checkpoint_path: str,
        device: Optional[str] = "auto",
        get_pooled_features: bool = True,
        **kwargs,
    ) -> "OptimizedReIDModel":
        """
        Create an `OptimizedReIDModel` with a [timm](https://huggingface.co/docs/timm)
        model as the backbone.

        Args:
            model_name_or_checkpoint_path (str): Name of the timm model to use or
                path to a safetensors checkpoint. If the exact model name is not
                found, the closest match from `timm.list_models` will be used.
            device (str): Device to run the model on.
            get_pooled_features (bool): Whether to get the pooled features from the
                model or not.
            **kwargs: Additional keyword arguments to pass to
                [`timm.create_model`](https://huggingface.co/docs/timm/en/reference/models#timm.create_model).

        Returns:
            OptimizedReIDModel: A new instance of `OptimizedReIDModel`.
        """

        # Note: This is a bit of a hack to get the type checker to work. It doesn't realize that calling
        # super().from_timm will return an OptimizedReIDModel. This gets around that.

        # Call the parent's from_timm method, which will create the correct subclass instance
        # thanks to the cls parameter being passed to the helper functions.
        return super().from_timm(model_name_or_checkpoint_path, device, get_pooled_features, **kwargs)

    def compile_and_warmup(self):
        """
        Compiles the model and warms it up. This might take a few seconds.
        """
        # no op if already compiled or compilation is not supported
        if self.compiled or not self.can_batch:
            return

        self.backbone_model = torch.compile(self.backbone_model)

        # we warm up the model with various batch sizes to hopefully hit all of the JIT compilation paths we would
        # expect to see at inference time.
        for bs in range(10):
            self.backbone_model(torch.randn(bs, 3, 224, 224).to(self.device))
        self.compiled = True

    def extract_features(self, detections: sv.Detections, frame: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Extract features from detection crops, using batched inference if self.can_batch.

        Args:
            detections (sv.Detections): Detections from which to extract features.
            frame (np.ndarray or PIL.Image.Image): The input frame.

        Returns:
            np.ndarray: Extracted features for each detection.
        """

        if self.can_batch:
            return self._extract_features_batched(detections, frame)

        return super().extract_features(detections, frame)

    def _extract_features_batched(self, detections: sv.Detections, frame: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Args:
            detections (sv.Detections): Detections from which to extract features.
            frame (np.ndarray or PIL.Image.Image): The input frame.

        Returns:
            np.ndarray: Extracted features for each detection.
        """
        if len(detections) == 0:
            return np.array([])

        if isinstance(frame, Image.Image):
            frame = np.array(frame)

        crops = []
        for box in detections.xyxy:
            crop = sv.crop_image(image=frame, xyxy=[*box.astype(int)])
            tensor = self.inference_transforms(crop)  # [C, H, W]
            crops.append(tensor)

        batch_tensor = torch.stack(crops).to(self.device)  # [N, C, H, W], where N = len(detections)

        with torch.inference_mode():
            batch_features = self.backbone_model(batch_tensor)  # [N, feature_dim]

        return batch_features.cpu().numpy()


class DeepSORTTracker(SORTTracker):
    """
    DeepSORT tracker for use with the Groundlight Python SDK. Inherits from SORTTracker
    and extends its functionality.

    Under the hood, we use the DeepSORT implementation from the trackers package.

    Example usage:
    ```python
    from groundlight import Groundlight
    from tracking.deepsort import DeepSORTTracker

    gl = Groundlight()
    object_detector = gl.create_counting_detector(
        name="people_counter",
        query="how many people are in the image?",
        class_names=["person"],
        max_count=10,
        confidence_threshold=0.75,
        patience_time=30.0,
    )


    tracker = DeepSORTTracker(
        image_width=1024,
        image_height=1024,
        classes_to_track=["person"],
        model_name="mobilenetv4_conv_small.e1200_r224_in1k",
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
        model_name: str = "mobilenetv4_conv_small.e1200_r224_in1k",
        classes_to_track: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Initializes the tracker.

        :param classes_to_track: A list of classes to track. If None, all classes will be tracked. If not None, the
            tracker will prune detections to only include these classes.
        :param image_width: Width of the input images in pixels.
        :param image_height: Height of the input images in pixels.
        :param model_name: Name of model from timm to extract appearance features. Default is a MobileNet-V4 image
            classification model. See https://huggingface.co/timm for a full list of available models. Some options
            include:
            - mobilenetv4_conv_small.e1200_r224_in1k (default): 3.8M params,
                https://huggingface.co/timm/mobilenetv4_conv_small.e1200_r224_in1k
            - mobilenetv3_small_100.lamb_in1k: 2.5M params, https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k
            - resnet50.a1_in1k: 25.6M params, https://huggingface.co/timm/resnet50.a1_in1k
            - mobilenetv3_large_100.ra_in1k: 5.5M params, https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k

        :param kwargs: Additional configuration parameters for the internal DeepSORT tracker.
            See https://trackers.roboflow.com/develop/trackers/core/deepsort/tracker/ for a full list of available
            parameters

            At the time of writing, the available parameters are:
            - lost_track_buffer: int (default: 30) - Number of frames to buffer when a track is lost
            - frame_rate: float (default: 30.0) - Frame rate of the video (frames per second)
            - track_activation_threshold: float (default: 0.25) - Detection confidence threshold for track activation
            - minimum_consecutive_frames: int (default: 3) - Number of consecutive frames that an object must be tracked
                before it is considered valid
            - minimum_iou_threshold: float (default: 0.3) - IOU threshold for associating detections to existing tracks
            - appearance_threshold: float (default: 0.7) - Cosine distance threshold for appearance matching.
                only matches below this threshold are considered valid.
            - appearance_weight: float (default: 0.5) - Weight (0-1) balancing motion (IOU) and appearance distance
                in the combined matching cost
            - distance_metric: str (default: 'cosine') - Distance metric for appearance features (e.g. 'cosine',
                'euclidean'). See 'scipy.spatial.distance.cdist'.
        """
        super().__init__(image_width=image_width, image_height=image_height, classes_to_track=classes_to_track)

        reid_model = OptimizedReIDModel.from_timm(model_name)
        reid_model.compile_and_warmup()

        self._internal_tracker = RoboflowDeepSORTTracker(reid_model, **kwargs)

    def _update_common(self, rois: list[ROI], frame: ImageType) -> None:
        """
        Internal common method that handles the common tracking logic for both update methods.
        """
        rois = self._filter_to_classes_to_track(rois)
        detections = self._rois_to_detections(rois)

        self._current_tracks = self._internal_tracker.update(detections=detections, frame=frame)

        # Update class cache for matched tracks
        if self._current_tracks is not None and len(self._current_tracks) > 0:
            if self._current_tracks.tracker_id is not None and self._current_tracks.class_id is not None:
                for track_id, class_id in zip(self._current_tracks.tracker_id, self._current_tracks.class_id):
                    if track_id != -1:
                        self._track_id_to_class_id[int(track_id)] = class_id
