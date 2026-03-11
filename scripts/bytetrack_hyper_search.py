"""
Performs a hyperparameter search over key hyperparameters for ByteTrack
"""

import copy

import numpy as np
from hyperparameter_search import run_hyperparameter_search

from test.unit.mock_provider import create_mock_sdk_provider
from tracking.bytetrack import ByteTrackTracker
from tracking.evaluation_utils import evaluate_tracking_performance
from tracking.track import Track


def evaluate_hyperparameters(hyper_dict):
    example_dataset = hyper_dict['example_dataset']
    dataset, image_width, image_height = create_mock_sdk_provider(
        return_mode="roi", detections_mode="ml_inferences", example_name=example_dataset
    )
    tracker = ByteTrackTracker(
        image_width=image_width,
        image_height=image_height,
        frame_rate=30,
        lost_track_buffer=hyper_dict['lost_track_buffer'],
        minimum_matching_threshold=hyper_dict['minimum_matching_threshold'],
        track_activation_threshold=hyper_dict['track_activation_threshold'],
        minimum_consecutive_frames=hyper_dict['mininum_consecutive_frames']
    )
    tracks_by_frame: dict[int, list[Track]] = {}

    for image, data, _ in dataset:
        tracker.update_from_rois(data, np.array(image))
        tracks = tracker.get_tracks()
        tracks_by_frame[len(tracks_by_frame)] = tracks
    
    metrics = evaluate_tracking_performance(
        dataset=dataset,
        predicted_tracks=tracks_by_frame,
        return_mode="roi",
        metrics=[
            "mota",
            "num_switches",
            "mostly_tracked",
            "partially_tracked",
            "mostly_lost",
        ],
    )
    metric_dict = copy.deepcopy(hyper_dict)

    metric_dict['mota'] = float(metrics['mota'])
    metric_dict['num_switches'] = int(metrics['num_switches'])
    metric_dict['mostly_tracked'] = int(metrics['mostly_tracked'])
    metric_dict['partially_tracked'] = int(metrics['partially_tracked'])
    metric_dict['mostly_lost'] = int(metrics['mostly_lost'])

    return metric_dict


def main():
    example_dataset_names = ['volleyball_example', 'person_example']
    track_activation_thresholds = [0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1]
    lost_track_buffers = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    minimum_matching_thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    min_consecutive_frames = [1, 5, 10, 15, 20, 25, 30, 45, 60]
    
    hyper_configs = []

    for example_dataset in example_dataset_names:
        for track_activation_threshold in track_activation_thresholds:
            for lost_track_buffer in lost_track_buffers:
                for minimum_matching_threshold in minimum_matching_thresholds:
                    for minimum_consecutive_frames in min_consecutive_frames:
                        hyper_dict = {
                            "example_dataset": example_dataset,
                            'track_activation_threshold': track_activation_threshold,
                            'lost_track_buffer': lost_track_buffer,
                            'minimum_matching_threshold': minimum_matching_threshold,
                            'mininum_consecutive_frames': minimum_consecutive_frames,
                        }
                        hyper_configs.append(hyper_dict)
    
    run_hyperparameter_search(evaluate_hyperparameters, hyper_configs)

if __name__ == '__main__':
    main()