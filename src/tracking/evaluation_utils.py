from typing import Union

import motmetrics as mm
import numpy as np
from groundlight import ROI

from tracking.track import Track


def convert_tracks_to_motmetrics_format(tracks: list[Track]) -> np.ndarray:
    """
    Convert Track objects to motmetrics format [x_center, y_center, width, height].

    Args:
        tracks: List of Track objects with normalized bbox coordinates

    Returns:
        np.ndarray: Array of shape (N, 4) in motmetrics format with normalized coordinates
    """
    if not tracks:
        return np.array([]).reshape(0, 4)

    formatted_tracks = []
    for track in tracks:
        # track.bbox is in normalized coordinates [x1, y1, x2, y2]
        x1_norm, y1_norm, x2_norm, y2_norm = track.bbox
        # Convert to center + width/height format (staying in normalized coordinates)
        x_center = (x1_norm + x2_norm) / 2
        y_center = (y1_norm + y2_norm) / 2
        width = x2_norm - x1_norm
        height = y2_norm - y1_norm
        formatted_tracks.append([x_center, y_center, width, height])

    return np.array(formatted_tracks)


def convert_rois_to_motmetrics_format(rois: list[ROI]) -> np.ndarray:
    """
    Convert ROI objects to motmetrics format [x_center, y_center, width, height].

    Args:
        rois: List of ROI objects with normalized coordinates

    Returns:
        np.ndarray: Array of shape (N, 4) in motmetrics format
    """
    if not rois:
        return np.array([]).reshape(0, 4)

    formatted_rois = []
    for roi in rois:
        box = roi.geometry
        width = box.right - box.left
        height = box.bottom - box.top
        formatted_rois.append([box.x, box.y, width, height])

    return np.array(formatted_rois)


def evaluate_tracking_performance(  # pylint: disable=too-many-locals
    *,
    dataset,
    predicted_tracks: dict[int, list[Track]],
    return_mode: str,
    metrics: list[str],
) -> dict[str, float]:
    """
    Evaluate tracking performance using motmetrics.

    Args:
        dataset: Dataset iterator yielding (image, ground_truth_data) tuples
        predicted_tracks: Dictionary mapping frame_idx to list of Track objects
        return_mode: Either "imagequery" or "roi" to determine how to extract ground truth
        metrics: List of metric names to compute. See https://github.com/cheind/py-motmetrics for available metrics.

    Returns:
        dict[str, float]: Dictionary mapping metric names to their computed values
    """
    if len(metrics) == 0:
        raise ValueError("metrics list cannot be empty")

    accumulator = mm.MOTAccumulator(auto_id=True)

    for frame_idx, (_, data, metadata) in enumerate(dataset):
        # Get ground truth data for this frame
        if return_mode == "imagequery":
            ground_truth_rois = data.rois if data.rois else []
        else:
            ground_truth_rois = data

        # Get predicted tracks for this frame
        predicted_tracks_for_frame = predicted_tracks.get(frame_idx, [])

        # Convert to motmetrics format
        gt_formatted = convert_rois_to_motmetrics_format(ground_truth_rois)
        predicted_formatted = convert_tracks_to_motmetrics_format(predicted_tracks_for_frame)

        # Calculate distance matrix (1-IOU)
        iou_cost_matrix = mm.distances.iou_matrix(gt_formatted, predicted_formatted, max_iou=1.0)

        # Object IDs - if not included in the metadata, we use the index of the ROI
        gt_object_ids = [metadata[i].get("track_id", i) for i in range(len(ground_truth_rois))]
        predicted_object_ids = [track.id for track in predicted_tracks_for_frame]

        # Update accumulator
        accumulator.update(gt_object_ids, predicted_object_ids, iou_cost_matrix)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(accumulator, metrics=metrics + ["num_frames"], name="acc")

    # Extract metrics into dictionary
    results = {}
    for metric in metrics:
        results[metric] = summary.loc["acc", metric]

    return results


def assert_metrics(
    metrics: dict[str, float], expected_values: dict[str, Union[float, tuple[float, float]]], test_description: str = ""
) -> None:
    """
    Assert that computed metrics meet expected thresholds.

    Args:
        metrics: Dictionary of computed metric values
        expected_values: Dictionary of expected thresholds with metric names as keys.
                        Values can be:
                        - float: minimum threshold (metric >= value)
                        - tuple[float, float]: (min, max) range (min <= metric <= max)
                                              If min == max, treated as exact value
        test_description: Optional description for better error messages
    """
    context = f" for {test_description}" if test_description else ""

    for metric_name, expected_value in expected_values.items():
        if metric_name not in metrics:
            raise ValueError(f"Metric '{metric_name}' not found in computed metrics")

        actual_value = metrics[metric_name]

        if isinstance(expected_value, tuple):
            min_val, max_val = expected_value
            if min_val == max_val:
                # Exact value check
                assert (
                    actual_value == min_val
                ), f"{metric_name.upper()} should be exactly {min_val}{context}, got {actual_value}"
            else:
                # Range check
                assert (
                    min_val <= actual_value <= max_val
                ), f"{metric_name.upper()} should be between {min_val} and {max_val}{context}, got {actual_value}"
        else:
            # Single value - minimum threshold
            # this assert satisfies the type checker for the comparison we actually care about
            assert isinstance(expected_value, (float, int))

            assert (
                actual_value >= expected_value
            ), f"{metric_name.upper()} should be >= {expected_value}{context}, got {actual_value}"


def exact(value: float) -> tuple[float, float]:
    """Helper to create exact value assertions."""
    return (value, value)
