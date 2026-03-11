import json
import os

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(script_dir, "img")
detections_path = os.path.join(script_dir, "detections.jsonl")

image_paths = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".jpg")])

# for each image, we need to draw the relevant detections on it. So first we need to read the detections.jsonl file and
# organize the detections by frame index -> list of detections
detections: dict[int, list[dict]] = {}

with open(detections_path, "r") as f:
    for line in f:
        data = json.loads(line)
        frame_index = data["frame_index"]

        if frame_index not in detections:
            detections[frame_index] = []

        detections[frame_index].append(data)

track_box_annotator = sv.BoxAnnotator(thickness=2)

class_ids = {
    "sports ball": 0,
    "person": 1,
}

# Store all annotated images for video creation
annotated_images = []


for frame_index, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Creating video"):
    detections_for_frame = detections[frame_index]
    image = Image.open(image_path)

    width, height = image.size

    # convert detections to sv.Detections
    xyxy = np.array([detection["box"] for detection in detections_for_frame])

    # scale the boxes to the image size
    xyxy[:, 0] = xyxy[:, 0] * width
    xyxy[:, 1] = xyxy[:, 1] * height
    xyxy[:, 2] = xyxy[:, 2] * width
    xyxy[:, 3] = xyxy[:, 3] * height

    confidence = np.array([detection["confidence"] for detection in detections_for_frame])
    class_id = np.array([class_ids[detection["class_name"]] for detection in detections_for_frame])
    sv_detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
    )
    annotated_image = track_box_annotator.annotate(scene=image, detections=sv_detections)

    # Convert PIL Image to OpenCV format (RGB to BGR)
    annotated_image_cv = cv2.cvtColor(np.array(annotated_image), cv2.COLOR_RGB2BGR)
    annotated_images.append(annotated_image_cv)

# Create video from annotated images
video_path = os.path.join(script_dir, "volleyball_gt_video.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
fps = 24  # frames per second

# Use the dimensions from the first image
height, width = annotated_images[0].shape[:2]
writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

try:
    for annotated_image in annotated_images:
        writer.write(annotated_image)
    print(f"Video saved to: {video_path}")
finally:
    writer.release()
