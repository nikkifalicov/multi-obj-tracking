"""
This script collect ROIs from the GL service for this video and saves the result to a jsonl file so we can use the
results to test trackers.
"""

import json
import os
import random

import tqdm
from groundlight import Groundlight

gl = Groundlight()

# this is a detector on our prod tracking canary account (see 1Password)
detector_id = "det_2yxfeL645d3wzVP7cuDpSVwv5rC"

# this detector is a counting detector with a confidence threshold of 0.75 and cloud labeling disabled
object_detector = gl.get_detector(detector_id)

# collect all image paths
images_path = "test/assets/person_example/img"
image_paths = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".jpg")])

# check if we should warm up the detector based on user input
warm_up = input("Warm up the detector? (y/n): ")
if warm_up == "y":
    random_image_paths = random.sample(image_paths, 30)
    for image_path in tqdm.tqdm(random_image_paths, desc="Warming up detector"):
        iq = gl.ask_async(object_detector, image_path)

# check if we should send all images to the detector
send_all = input("Send all images to the detector? (y/n): ")
if send_all == "y":
    for index, image_path in enumerate(tqdm.tqdm(image_paths, desc="Sending all images to detector")):
        iq = gl.ask_async(object_detector, image_path, metadata={"frame_index": index})


# collect all the IQs
print("Collecting all IQs on the detector")
all_queries = []
page = 1
page_size = 1000
while True:
    try:
        queries = gl.list_image_queries(page=page, page_size=page_size, detector_id=detector_id)
        all_queries.extend(queries.results)
        if len(queries.results) < page_size:  # If we got fewer than the page size, we're likely at the end
            break
        page += 1
    except Exception as e:
        if "404" in str(e) or "Invalid page" in str(e):
            break
        else:
            raise  # Re-raise if it's a different error

print(f"Collected {len(all_queries)} total queries on the detector")

# filter to only those with metadata
filtered_queries = [iq for iq in all_queries if iq.metadata is not None and "frame_index" in iq.metadata]
print(f"Collected {len(filtered_queries)} total queries on the detector with metadata")

# sort them from lowest to highest frame index
sorted_queries = sorted(filtered_queries, key=lambda x: x.metadata["frame_index"])

# save the ROIs to a jsonl file
with open("test/assets/person_example/ml_inferences.jsonl", "w", encoding="utf-8") as f:
    for query in sorted_queries:
        frame_index = query.metadata["frame_index"]
        rois = query.rois
        if rois is not None and len(rois) > 0:
            for roi in rois:
                # write the roi to the jsonl file
                data = {
                    "frame_index": int(frame_index),
                    "class_name": roi.label,
                    "box": [roi.geometry.left, roi.geometry.top, roi.geometry.right, roi.geometry.bottom],
                    "confidence": roi.score,
                }
                f.write(json.dumps(data) + "\n")
print("Wrote ROIs to ml_inferences.jsonl")
