This directory contains an example for object tracking tests. It is from the LaSOT dataset: http://vision.cs.stonybrook.edu/~lasot/index.html. This example is `person-12`, in which we are tracking a single object, a gymnast, from a fixed camera view.

Contents:

- `img/` - contains the frames from the video as jpg frames.
- `groundtruth.txt` - contains the ground truth bounding boxes for the object in each frame
- `full_occlusion.txt` - array that is 1 if the object is fully occluded in the frame, 0 otherwise
- `out_of_view.txt` - array that is 1 if the object is out of view in the frame, 0 otherwise
- `video.mp4` - the original video
- `detections.jsonl` - contains the ground truth detections for the object in each frame (derived from groundtruth.txt)
- `ml_inferences.jsonl` - contains the ML inferences for each frame (collected using the Groundlight service and `collect_gl_inferences.py`)


