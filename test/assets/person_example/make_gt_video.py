# converts the frames in /img + groundtruth.txt to a video with the annotations

import os

import cv2
import numpy as np

# Get all jpg files in the img directory and sort them
img_files = sorted([f for f in os.listdir("img") if f.endswith(".jpg")])

# Read the frames using the actual filenames
frames = []
for img_file in img_files:
    frame = cv2.imread(f"img/{img_file}")
    if frame is not None:
        frames.append(frame)

print(f"Loaded {len(frames)} frames")

# Read the groundtruth.txt file
groundtruth = np.loadtxt("groundtruth.txt", delimiter=",")

# Check if we have matching number of frames and groundtruth entries
print(f"Groundtruth entries: {len(groundtruth)}")

# Use the minimum of frames and groundtruth entries to avoid index errors
min_length = min(len(frames), len(groundtruth))

# Map the groundtruth to the frames and draw the bounding boxes
for i in range(min_length):
    frame = frames[i]
    cv2.rectangle(
        frame,
        (int(groundtruth[i, 0]), int(groundtruth[i, 1])),
        (int(groundtruth[i, 0] + groundtruth[i, 2]), int(groundtruth[i, 1] + groundtruth[i, 3])),
        (0, 0, 255),
        2,
    )

# Create video only if we have frames
if frames:
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    video = cv2.VideoWriter("groundtruth.mp4", fourcc, 30.0, (width, height))

    for frame in frames[:min_length]:
        video.write(frame)

    video.release()
    print("Video saved as groundtruth.mp4")
else:
    print("No frames loaded!")
