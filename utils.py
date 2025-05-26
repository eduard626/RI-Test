import numpy as np
from typing import Union
import cv2


def compute_iou(
    boxA: Union[tuple, list],
    boxB: Union[tuple, list],
    overlap_only: bool = False,
) -> float:
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Optionally, it can return only the proportion of overlap of the intersection wrt boxA.

    Args:
        boxA (Union[tuple, list]): The first (reference) bounding box in the format (x1, y1, x2, y2).
        boxB (Union[tuple, list]): The second (candidate) bounding box in the format (x1, y1, x2, y2).
        overlap_only (bool, optional): If True, returns only the proportion of overlap of the intersection wrt boxA. Defaults to False.

    Returns:
        float: The IoU between the two bounding boxes. If overlap_only is True, returns the proportion of overlap of the intersection wrt boxA.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if overlap_only:
        # return only the proportion of overlap of the intersection wrt boxA
        if boxAArea == 0:
            return 0.0
        return interArea / float(boxAArea)
    return interArea / float(
        boxAArea + boxBArea - interArea + 1e-6
    )  # Add a small value to avoid division by zero


def background_subtraction(
    background_frame: np.ndarray, current_frame: np.ndarray, threshold: int = 30
) -> np.ndarray:
    """
    Applies background subtraction to the current frame using the background frame.
    This function computes the absolute difference between the background and current frames,
    applies a Gaussian blur, and then thresholds the result to create a foreground mask.

    Args:
        background_frame (np.ndarray): The background frame to compare against.
        current_frame (np.ndarray): The current frame to process.
        threshold (int, optional): The threshold value for the absolute difference. Defaults to 30.

    Raises:
        ValueError: If the background and current frames do not have the same shape.

    Returns:
        np.ndarray: The foreground mask after applying background subtraction.
    """

    if background_frame.shape != current_frame.shape:
        raise ValueError("Background and current frame must have the same shape.")
    current_frame = cv2.GaussianBlur(current_frame, (7, 7), 0)
    diff = cv2.absdiff(background_frame, current_frame)
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    return fg_mask


def find_segments(mask: np.ndarray, min_area: int = 500, as_rect: bool = True) -> list:
    """
    Finds segments in a binary mask using contour detection.

    Args:
        mask (np.ndarray): Binary mask where segments are to be detected.
        min_area (int, optional): Minimum area of a segment to be considered valid. Defaults to 500.
        as_rect (bool, optional): If True, returns segments as rectangles (x, y, w, h). If False, returns contours. Defaults to True.

    Returns:
        list: List of segments found in the mask, each represented as a tuple (x, y, w, h) or a contour.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segments = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            if not as_rect:
                segments.append(contour)
                continue
            x, y, w, h = cv2.boundingRect(contour)
            segments.append((x, y, w, h))
    return segments


def filter_boxes_fg(
    boxes: np.ndarray,
    fg_blobs: list,
    min_overlap: float = 0.25,
    overlap_only: bool = False,
) -> list:
    """
    Filters boxes based on foreground blobs.
    Only keeps boxes that overlap with at least one foreground blob by a certain percentage.

    Args:
        boxes (np.ndarray): Array of boxes in xywh format.
        fg_blobs (list): List of foreground blobs in xywh format.
        min_overlap (float): Minimum overlap percentage to keep the box.

    Returns:
        list: Filtered boxes that overlap with foreground blobs.
    """
    filtered_boxes = []
    for i, box in enumerate(boxes):
        x, y, w, h = map(int, box)
        has_overlap = False
        for blob in fg_blobs:
            blob_x, blob_y, blob_w, blob_h = blob
            iou = compute_iou(
                (x, y, x + w, y + h),
                (blob_x, blob_y, blob_x + blob_w, blob_y + blob_h),
                overlap_only=overlap_only,
            )
            if iou > min_overlap:  # min overlap
                has_overlap = True
                break
        if has_overlap:
            filtered_boxes.append(i)
    return filtered_boxes
