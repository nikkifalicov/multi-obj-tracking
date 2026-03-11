import cv2
import numpy as np
import supervision as sv

from src.tracking import bytetrack, deepsort
from test.unit import mock_provider
from tracking.evaluation_utils import (
    evaluate_tracking_performance,
)
from tracking.track import Track

# change if you do not want output to be saved
should_save_video = True
video_path = "bytetrack_tracker_on_gl_inferences_demo.mp4"

# initialize mock SDK provider to simulate responses from Groundlight
dataset, image_width, image_height = mock_provider.create_mock_sdk_provider(
    return_mode="roi", detections_mode="ml_inferences", example_name="person_example"
)

# initialize ByteTrackTracker
tracker = bytetrack.ByteTrackTracker(
    image_width=image_width,
    image_height=image_height,
    minimum_matching_threshold=0.9,
    minimum_consecutive_frames=5,
)

# # initialize DeepSORT Tracker (if GPU available)
# tracker = deepsort.DeepSORTTracker(
#     image_width=image_width,
#     image_height=image_height,
#     model_name = "mobilenetv4_conv_small.e1200_r224_in1k"
# )

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

    # annotate the image with the raw detections, if present
    raw_detections = tracker._rois_to_detections(data)
    annotated_image = detection_box_annotator.annotate(
        image_cv, raw_detections)

    if len(data) == 0:
        # write no detections to the annotated image in the bottom right corner in white
        cv2.putText(annotated_image, "No detections", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(video_path, fourcc, 30,
                             (image_width, image_height))
    try:
        for annotated_image in annotated_images:
            writer.write(annotated_image)
    finally:
        writer.release()

# Evaluate tracking performance using motmetrics against ground truth
ground_truth_dataset, _, _ = mock_provider.create_mock_sdk_provider(
    return_mode="roi", detections_mode="groundtruth", example_name="person_example"
)

metrics = evaluate_tracking_performance(
    dataset=ground_truth_dataset,
    predicted_tracks=tracks_by_frame,
    return_mode="roi",
    metrics=["mota"],
)

print(metrics)
