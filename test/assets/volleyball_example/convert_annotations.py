"""
Converts annotations.xml (CVAT for video 1.1 format) to a JSONL file for our testing purposes.

The annotations.xml file was created by labeling this example with the CVAT GUI application.

See format spec here: https://docs.cvat.ai/docs/manual/advanced/xml_format/
"""

import json
import os
import xml.etree.ElementTree as ET

import cv2

from tracking.bbox_utils import pixels_to_normalized

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_dir, "annotations.xml")
jsonl_path = os.path.join(script_dir, "detections.jsonl")

frame_path = os.path.join(script_dir, "img/00000001.jpg")


def parse_cvat_tracks(xml_file_path):
    """
    Parse CVAT XML file and extract track detections.

    Returns:
        list: List of detections with track info
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # get the video's width and height to normalize the bounding boxes
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape

    detections = []

    # Iterate over each track
    for track in root.findall("track"):
        track_id = int(track.get("id", "0"))
        track_label = track.get("label", "")

        # Get all boxes for this track
        for box in track.findall("box"):
            x1_pixel = float(box.get("xtl", "0"))
            y1_pixel = float(box.get("ytl", "0"))
            x2_pixel = float(box.get("xbr", "0"))
            y2_pixel = float(box.get("ybr", "0"))

            bounding_box = pixels_to_normalized(
                x1=x1_pixel, y1=y1_pixel, x2=x2_pixel, y2=y2_pixel, image_width=width, image_height=height
            )

            detection = {
                "track_id": track_id,
                "class_name": track_label,
                "frame_index": int(box.get("frame", "0")),
                "box": bounding_box,
                "confidence": 1.0,
            }
            detections.append(detection)

    return detections


def convert_to_jsonl(detections):
    """
    Convert detections to JSONL format.
    """
    with open(jsonl_path, "w") as f:
        for det in detections:
            f.write(json.dumps(det) + "\n")


def main():
    print(f"Parsing tracks from: {xml_path}")

    detections = parse_cvat_tracks(xml_path)

    print(f"Found {len(detections)} detections")

    # Show a few examples
    for i, det in enumerate(detections[:5]):
        print(det)

    convert_to_jsonl(detections)


if __name__ == "__main__":
    main()
