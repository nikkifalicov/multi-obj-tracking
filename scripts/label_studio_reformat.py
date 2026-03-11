import json
import os
import argparse
import numpy as np
import pandas as pd


def linear_interpolate(start: int, end: int, num_frames: int) -> list:
    """
    Perform linear interpolation between two values

    Args:
        start (int): starting value for interpolation
        end (int): ending value for interpolation
        num_frames (int): number of frames between the value at start and end
    Returns:
        list of interpolated values between the provided start and end
    """
    num_frames = int(num_frames)
    vals = [start + ((end - start) * (i/num_frames))
            for i in range(1, num_frames)]
    vals = [int(x) for x in vals]
    return vals


def interpolate_track(detections_data: dict, id: int) -> dict:
    """
    Performs linear interpolation between all keypoints with interpolation
    enabled for a single object's track

    Args:
        detections_data (dict): dictionary of keyframe detection info
        id (int): numeric id for the current object
    Returns:
        dict containing all information for the interpolated frames. Does
        not include the information from the original frames
    """
    # information about keys of interest
    columns_to_interpolate = ['bb_left', 'bb_top', 'bb_width', 'bb_height']
    interpolation_results = {
        "frame": [],
        "id": [],
        "bb_left": [],
        "bb_top": [],
        "bb_width": [],
        "bb_height": [],
        "conf": [],
        "x": [],
        "y": [],
        "z": [],
        "interp_enabled": []
    }

    # we only want to do interpolation between the keyframes with the same id
    # as the one that we are interested in
    indices = np.where(np.array(detections_data['id']) == id)[0].tolist()

    # iterate through all keyframes with the current id, ignoring last keyframe
    for i, keyframe_idx in enumerate(indices[:-1]):

        # only interpolate if it is enabled for the keyframe
        if detections_data['interp_enabled'][keyframe_idx]:
            start_frame = detections_data['frame'][keyframe_idx]
            end_frame = detections_data['frame'][indices[i+1]]
            frames_interpolated = end_frame-start_frame-1

            interpolation_results['frame'] += [
                x for x in range(start_frame+1, end_frame)]
            interpolation_results['id'] += [id]*frames_interpolated
            interpolation_results['conf'] += [1]*frames_interpolated
            interpolation_results['x'] += [0]*frames_interpolated
            interpolation_results['y'] += [0]*frames_interpolated
            interpolation_results['z'] += [0]*frames_interpolated

            # perform interpolation on frames of interest
            for value in columns_to_interpolate:
                start_val = detections_data[value][keyframe_idx]
                end_val = detections_data[value][indices[i+1]]
                interpolated = linear_interpolate(
                    start_val, end_val, end_frame-start_frame)
                interpolation_results[value] += interpolated

    interpolation_results['interp_enabled'] += [False] * \
        (len(interpolation_results['frame']))
    return interpolation_results


def process_single_json(args, json_name, json_data):
    """
    Transforms the data from a single Label Studio JSON

    Args:
        args (Namespace): parsed command-line arguments
        json_name (str): name of the .json file of interest relative to the
            input directory
        json_data (dict): parsed .json corresponding to json_name
    Returns:
        No returns, but outputs the .json information to the output path
        specified by args with the same name as the original .json
    """
    # dictionary to store detection info
    detections_data = {
        "frame": [],
        "id": [],
        "bb_left": [],
        "bb_top": [],
        "bb_width": [],
        "bb_height": [],
        "conf": [],
        "x": [],
        "y": [],
        "z": [],
        "interp_enabled": []
    }

    num_ids = 0
    seen_ids = {}
    height = args.img_height
    width = args.img_width

    # iterate through all tracks
    all_track_data = json_data[0]['annotations'][0]['result']
    for track_data in all_track_data:
        id = track_data['id']

        # assign a unique id to each object
        if id not in seen_ids:
            seen_ids[id] = num_ids
            id_numeric = num_ids
            num_ids += 1
        else:
            id_numeric = seen_ids[id]

        track_sequence = track_data['value']['sequence']

        # iterate through the sequence for the track
        for keypoint_idx, keypoint_frame_info in enumerate(track_sequence):

            # transform coordinates from normalized to pixel
            left = int(keypoint_frame_info['x']*width/100)
            top = int(keypoint_frame_info['y']*height/100)
            w = int(keypoint_frame_info['width']*width/100)
            h = int(keypoint_frame_info['height']*height/100)

            # update detections dictionary
            detections_data['frame'].append(keypoint_frame_info['frame'])
            detections_data['id'].append(id_numeric)
            detections_data['bb_left'].append(left)
            detections_data['bb_top'].append(top)
            detections_data['bb_width'].append(w)
            detections_data['bb_height'].append(h)
            detections_data['conf'].append(1)
            detections_data['x'].append(0)
            detections_data['y'].append(0)
            detections_data['z'].append(0)
            interp_enabled = track_sequence[keypoint_idx]['enabled']
            detections_data['interp_enabled'].append(interp_enabled)

    # if interpolation is enabled, perform interpolation
    if args.interpolate:
        for _, id in seen_ids.items():
            interpolation_results = interpolate_track(detections_data, id)
            for key, value in interpolation_results.items():
                detections_data[key] += value

    # save the detections dictionary as a text file
    df = pd.DataFrame.from_dict(detections_data)
    df = df.sort_values(by=['id', 'frame'])
    fname = json_name.split(".")[0].split("/")[-1]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_file_path = os.path.join(args.output_dir, f"{fname}.csv")
    if os.path.exists(output_file_path):
        raise FileExistsError(
            f"File with name {output_file_path} already exists")
    df = df.drop('interp_enabled', axis=1)
    df.to_csv(output_file_path, header=False, index=False)


def process_jsons(args):
    """
    Process all of the Label Studio .json files and turn them into MOT20
    format

    Args:
        args (Namespace): parsed command-line arguments
    Returns:
        No returns, just saves the MOT20 .json files at the location specified
        by args
    """
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError("Input directory does not exist")
    for json_file in os.listdir(args.input_dir):
        # only process the json files in the directory
        if ".json" in json_file:
            full_json_path = os.path.join(args.input_dir, json_file)
            with open(full_json_path, 'r') as file:
                json_data = json.load(file)
                process_single_json(args, json_file, json_data)


def main(args):
    process_jsons(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='label_studio_reformat',
        description='Reformat Label Studio JSON files to MOT20 format',
    )

    parser.add_argument("--input_dir", type=str, required=True,
                        help="Input directory of Label Studio JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for reformatted JSON files")
    parser.add_argument("--interpolate", type=bool, default=True,
                        help="Whether to perform linear interpolation between frames")
    parser.add_argument("--img_height", type=int, default=1080,
                        help="Frame height in pixels")
    parser.add_argument("--img_width", type=int, default=1920,
                        help="Frame width in pixels")

    args = parser.parse_args()
    main(args)
