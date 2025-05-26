#!/usr/bin/env python3
"""
A script to detect and track things/interactions in the RI video.
We use the GroundingDINO model for zero-shot object detection.
We prune detections based on the foreground mask
We track with a simple centroid tracker.
Hand-object interactions are detected by checking if the hand is close to the object for a certain number of frames.
"""

import cv2
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm

from detector import GdDetector
from tracker import CentroidTracker
from utils import background_subtraction, find_segments, filter_boxes_fg


def run(args) -> None:
    """
    Main function to run the detection and tracking on the video.

    Args:
        args (_type_): The command line arguments containing video path, output path, and other parameters.
    """
    # open the video file, basic approach
    # we could pass additional parameters
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Processing video: {args.video}")
    if not cap.isOpened():
        print(f"Failed to open video file: {args.video}")
        exit(1)

    # we process the frames at a fixed height, width is scaled accordingly
    # ideally a smaller resolution for faster processing
    H = args.height

    # read the first frame, this is used to initialize the background subtraction
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the video.")
        exit()
    # Pre process the frame: downscale, color conversion, and blurring
    first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    scale = H / first_frame_gray.shape[0]
    first_frame_gray = cv2.resize(first_frame_gray, (0, 0), fx=scale, fy=scale)

    # Process video frames
    frame_count = 0  # Frame counter
    sample_every_n_frames = 1  # Sample every n frames
    # Limit processing to a maximum number of frames?
    # mainly for testing purposes
    max_frames = -1  # -1 means process all frames

    # when visualizing, we use matplotlib to show the frames
    if args.visualize:
        plt.ion()
        _, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = None  # No visualization

    pbar = tqdm(total=max_frames, desc="Processing frames", unit="-frame")

    # Initialize the GroundingDINO detector
    detector = GdDetector()
    detector.load_model()

    # Initialize the Centroid Tracker
    tracker = CentroidTracker(
        distance_threshold=args.track_distance,
        smoothing=args.tracking_smoothing,
        max_disappeared=args.track_max_frames,
    )

    # Loop through the video frames
    processing_time = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        if frame_count % sample_every_n_frames != 0:
            frame_count += 1
            continue
        # preprocess
        time_start = cv2.getTickCount()
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.GaussianBlur(frame_rgb, (7, 7), 0)

        # Run inference on the current frame
        boxes, logits, phrases = detector.predict_np(
            image_np=frame_rgb,
            text_prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

        # Apply background subtraction
        mask = background_subtraction(first_frame_gray, frame_gray, threshold=30)
        # Find segments in the mask, i.e., objects in the foreground
        blobs = find_segments(mask, min_area=1000)
        # filter detections based on the foreground mask
        filtered_ids = filter_boxes_fg(
            boxes, blobs, min_overlap=0.25, overlap_only=True
        )
        boxes = boxes[filtered_ids]
        phrases = [phrases[i] for i in filtered_ids]
        # @TODO: we are not using score at the moment, but we could use it to filter detections
        # scores = logits.sigmoid().cpu().numpy()
        # scores = scores[filtered_ids]

        # Track the detected objects
        # @TODO: We are constantly going back and forth between box formats
        # We should reconcile a single format across the codebase
        detections = []
        for box, label in zip(boxes, phrases):
            x, y, w, h = map(int, box)
            x1, y1, x2, y2 = x, y, x + w, y + h
            detections.append(((x1, y1, x2, y2), label))
        detections = tracker.update(detections)
        processing_time = (cv2.getTickCount() - time_start) / cv2.getTickFrequency()

        # Draw the tracked objects
        for object_id, obj in detections.items():
            centroid = obj.centroid
            avg_width = obj.avg_width
            avg_height = obj.avg_height
            class_label = obj.label
            x1 = int(centroid[0] - avg_width / 2)
            y1 = int(centroid[1] - avg_height / 2)
            x2 = int(centroid[0] + avg_width / 2)
            y2 = int(centroid[1] + avg_height / 2)
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_rgb,
                f"{class_label}: {object_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        if args.visualize:
            # Show the mask
            ax.clear()
            ax.imshow(frame_rgb)
            ax.axis("off")
            ax.set_title(f"Frame {frame_count} - FPS: {(1/processing_time):.2f}s")
            plt.draw()
            plt.pause(1 / fps)  # Pause to update the plot

        # Save the frame as jpeg, we will write the video later
        output_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        output_frame = cv2.resize(output_frame, (0, 0), fx=1 / scale, fy=1 / scale)
        frame_path = args.output + f"/frame_{frame_count:04d}.jpg"
        cv2.imwrite(frame_path, output_frame)
        frame_count += 1
        if max_frames > 0 and frame_count >= (max_frames * sample_every_n_frames):
            break
        pbar.update(1)
    pbar.close()
    # Release video capture
    cap.release()
    # use the frames to write a video
    frame_files = sorted(
        [f for f in os.listdir(args.output) if f.endswith(".jpg")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )
    if len(frame_files) == 0:
        raise ValueError("No frames found in the output directory!!!")
    # Create a video writer
    first_frame = cv2.imread(os.path.join(args.output, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4
    out = cv2.VideoWriter(
        args.output + "/output_video.mp4", fourcc, fps, (width, height)
    )
    # Write each frame to the video
    print(f"Writing video to {args.output}/output_video.mp4")
    for frame_file in tqdm(frame_files, desc="Writing video frames"):
        frame_path = os.path.join(args.output, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
    out.release()  # Release the video writer
    print(f"Video saved to {args.output}/output_video.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot object detection with GroundingDINO"
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to the output video file"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize the detections as we go"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="orange hand.petri dish.bottle.blue lid.",
        help="GroundingDINO prompt to use for detection",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="Text threshold for GroundingDINO",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.35,
        help="Box threshold for GroundingDINO",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Height of use to resize the video frames. Width is scaled accordingly.",
    )
    parser.add_argument(
        "--track_max_frames",
        type=int,
        default=15,  # @TODO: based on the fPS of the video?
        help="Maximum number of frames to track an object before considering it lost",
    )
    parser.add_argument(
        "--track_distance",
        type=float,
        default=100.0,  # @TODO: based on the size/type of objects? e.g., hands move fast
        help="Maximum distance to consider an object tracked",
    )
    parser.add_argument(
        "--tracking_smoothing",
        type=float,
        default=0.9,
        help="Smoothing factor for tracking",
    )
    args = parser.parse_args()
    # basic things?
    if not os.path.exists(args.output):
        # the user creates the output directory if it does not exist
        raise FileNotFoundError(
            f"Output directory {args.output} does not exist. Please create it."
        )

    # Check if the video file exists
    if not os.path.exists(args.video):
        print(f"Video file {args.video} does not exist.")
        exit(1)

    run(args)
