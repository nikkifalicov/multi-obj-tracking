"""
This script creates and collects ROIs from the GL service for this video and saves the result to a jsonl file so we can
use the results to test trackers.
"""

import json
import os
import random

import tqdm
from groundlight import Groundlight, ImageQuery


def get_iqs(
    gl: Groundlight,
    detector_id: str,
) -> dict[int, ImageQuery]:
    """
    Returns all the ImageQueries for a given detector id in order of frame index (filters out those without
    appropriate metadata)
    """
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

    # filter out those without a frame index
    filtered_queries = [iq for iq in all_queries if iq.metadata is not None and "frame_index" in iq.metadata]

    # convert into a dictionary of frame index to ImageQuery
    iqs_by_frame = {iq.metadata["frame_index"]: iq for iq in filtered_queries}
    return iqs_by_frame


def iq_to_data(iq: ImageQuery) -> list[dict]:
    """
    Converts an ImageQuery to a list of dictionaries of data or returns an empty list if no data to write.
    """

    rois = iq.rois
    all_data = []
    if rois is not None and len(rois) > 0:
        for roi in rois:
            # write the roi to the jsonl file
            assert iq.metadata is not None and "frame_index" in iq.metadata, "Frame index is required"
            data = {
                "frame_index": int(iq.metadata["frame_index"]),
                "class_name": roi.label,
                "box": [roi.geometry.left, roi.geometry.top, roi.geometry.right, roi.geometry.bottom],
                "confidence": roi.score,
            }
            all_data.append(data)

    return all_data


def main():
    gl = Groundlight()
    person_detector = gl.get_detector(id="det_2zZ9CdKNULcbHYAseJfJJF72HO6")
    volleyball_detector = gl.get_detector(id="det_2z70voP5Dpw1GlL45M7I6FZQ4Ep")

    images_path = "test/assets/volleyball_example/img"
    image_paths = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith(".jpg")])

    # check if we should warm up the detectors based on user input
    warm_up = input("Warm up the detectors? (y/n): ")
    if warm_up == "y":
        random_image_paths = random.sample(image_paths, 30)
        for image_path in tqdm.tqdm(random_image_paths, desc="Warming up detectors"):
            person_iq = gl.ask_async(person_detector, image_path)
            volleyball_iq = gl.ask_async(volleyball_detector, image_path)

    # check if we should send all images to the detectors based on user input
    send_all = input("Send all images to the detectors? (y/n): ")
    if send_all == "y":
        for index, image_path in enumerate(tqdm.tqdm(image_paths, desc="Submitting images to detectors")):
            person_iq = gl.ask_async(person_detector, image_path, metadata={"frame_index": index})
            volleyball_iq = gl.ask_async(volleyball_detector, image_path, metadata={"frame_index": index})

    print("Collecting IQs from detectors")
    person_iqs = get_iqs(gl, person_detector.id)
    volleyball_iqs = get_iqs(gl, volleyball_detector.id)
    print(f"Collected {len(person_iqs)} person IQs")
    print(f"Collected {len(volleyball_iqs)} volleyball IQs")

    min_frame_index = int(min(person_iqs.keys(), volleyball_iqs.keys()))
    max_frame_index = int(max(person_iqs.keys(), volleyball_iqs.keys()))
    print(f"Min frame index: {min_frame_index}")
    print(f"Max frame index: {max_frame_index}")

    with open("test/assets/volleyball_example/ml_inferences.jsonl", "w", encoding="utf-8") as f:
        for frame_index in range(min_frame_index, max_frame_index + 1):
            if frame_index in person_iqs:
                person_iq = person_iqs[frame_index]
                data = iq_to_data(person_iq)
                for d in data:
                    f.write(json.dumps(d) + "\n")
            if frame_index in volleyball_iqs:
                volleyball_iq = volleyball_iqs[frame_index]
                data = iq_to_data(volleyball_iq)
                for d in data:
                    f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    main()
