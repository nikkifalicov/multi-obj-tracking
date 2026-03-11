This directory contains an example for object tracking tests. It is from the LaSOT dataset: http://vision.cs.stonybrook.edu/~lasot/index.html. This example is `volleyball-8`, in which we are tracking two people and two balls.

Contents:
- `img/` - contains the frames from the video as jpg frames.
- `annotations.xml` - contains the ground truth annotations for the video in CVAT for video 1.1 format
- `video.mp4` - the original video
- `detections.jsonl` - contains the ground truth detections for the object in each frame (derived from annotations.xml using `convert_annotations.py`)



