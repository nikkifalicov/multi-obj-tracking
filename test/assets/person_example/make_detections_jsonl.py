"""
Converts the groundtruth.txt file (provided by dataset) to detections.jsonl, to be used to mock data for testing
"""

import json

from PIL import Image

from tracking.bbox_utils import pixels_to_normalized

image_fp = "test/assets/person_example/img/00000001.jpg"
fp = "test/assets/person_example/groundtruth.txt"
output_fp = "test/assets/person_example/detections.jsonl"

frame = Image.open(image_fp)
width, height = frame.size

boxes = []

with open(fp) as file:
    for frame_index, row in enumerate(file):
        vals = row.split(",")
        vals[-1] = vals[-1].removesuffix("\n")

        # Convert string values to integers
        int_vals = [int(val) for val in vals]

        # convert from xywh to xyxy
        x1, y1, box_width, box_height = int_vals
        x2 = x1 + box_width
        y2 = y1 + box_height

        # normalize to 0 - 1
        normalized_box = pixels_to_normalized(x1=x1, y1=y1, x2=x2, y2=y2, image_width=width, image_height=height)

        boxes.append(
            {
                "frame_index": frame_index,
                "box": normalized_box,
                "class_name": "human",
                "confidence": 1.0,
            }
        )

with open(output_fp, mode="w") as output:
    for box in boxes:
        output.write(json.dumps(box) + "\n")
