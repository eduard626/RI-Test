from collections import defaultdict
import numpy as np
from scipy.spatial import distance
from typing import Union

from utils import compute_iou


class ObjectTrack:
    """
    A class to represent objects and their info being tracked in a video stream.
    @TODO: further specialise for containers e.g., beakers, petri dishes, etc.
    # So that we can for instance track capacity/volume, fill level, lids/caps on/off
    """

    def __init__(
        self,
        object_id: int,
        centroid: "tuple[int,int]",
        bbox: Union[tuple, list],
        class_label: str,
        smoothing: float = 0.5,
    ):
        """
        Initializes an ObjectTrack instance.

        Args:
            object_id (int): The unique identifier for the object, .e.g, counter
            centroid (tuple[int,int]): The centroid of the object in the format (x, y).
            bbox (Union[tuple, list]): The bounding box of the object in the format (x1, y1, x2, y2).
            class_label (str): The class label of the object, e.g., "beaker", "petri dish", etc.
            smoothing (float, optional): Smoothing factor for centroid updates. Defaults to 0.5.
        """
        self.id = object_id
        self.label = class_label
        self.bbox = bbox
        self.centroid = np.array(centroid, dtype=np.float32)
        self.smoothing = smoothing

        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        self.avg_width = width
        self.avg_height = height
        # Number of updates to the object
        self.num_updates = 1
        # Frames since the object was last seen
        self.disappeared = 0
        # e.g., number of times the object was interacted with, e.g., picked up, moved, etc.
        # perhaps time-based and proximity-based
        # self.interactions =0

    def update(
        self,
        new_centroid: "tuple[int,int]",
        new_bbox: Union[tuple, list],
        class_label: str,
    ) -> None:
        """
        Updates the object's properties

        Args:
            new_centroid (tuple[int,int]): The new centroid of the object in the format (x, y).
            new_bbox (Union[tuple, list]): The new bounding box of the object in the format (x1, y1, x2, y2).
            class_label (str): The class label of the object, e.g., "beaker", "petri dish", etc.
        """
        self.centroid = (
            self.smoothing * np.array(new_centroid, dtype=np.float32)
            + (1 - self.smoothing) * self.centroid
        )

        self.bbox = new_bbox
        self.label = class_label

        x1, y1, x2, y2 = new_bbox
        width = x2 - x1
        height = y2 - y1
        n = self.num_updates
        self.avg_width = (self.avg_width * n + width) / (n + 1)
        self.avg_height = (self.avg_height * n + height) / (n + 1)
        self.num_updates += 1

        self.disappeared = 0


class CentroidTracker:
    def __init__(
        self,
        max_disappeared: int = 15,
        distance_threshold: int = 50,
        smoothing: float = 0.5,
        nms_iou_threshold: float = 0.5,
    ):
        """
        Initializes the CentroidTracker.
        Detections can be jittery or noisy, so we use a smoothing factor to average the centroid positions over time.
        Grounding DINO can return multiple detections for the same object, so we apply Non-Maximum Suppression (NMS) to filter out overlapping boxes.

        Args:
            max_disappeared (int, optional): The number of frames an object can be missing before it is deregistered. Defaults to 15.
            distance_threshold (int, optional): The maximum distance between centroids to consider them the same object. Defaults to 50.
            smoothing (float, optional): Smoothing factor for centroid updates. Defaults to 0.5.
            nms_iou_threshold (float, optional): The IoU threshold for Non-Maximum Suppression (NMS). Defaults to 0.5.
        """
        # per object id
        self.next_object_id = 0
        # collection of ObjectTrack instances
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold
        self.smoothing = smoothing
        self.nms_iou_threshold = nms_iou_threshold

    def _register(
        self, centroid: "tuple[int,int]", bbox: Union[tuple, list], class_label: str
    ) -> None:
        """
        Registers a new object with the tracker.

        Args:
            centroid (tuple[int,int]): The centroid of the object in the format (x, y).
            bbox (Union[tuple, list]): The bounding box of the object in the format (x1, y1, x2, y2).
            class_label (str): The class label of the object, e.g., "beaker", "petri dish", etc.
        """
        obj = ObjectTrack(
            object_id=self.next_object_id,
            centroid=centroid,
            bbox=bbox,
            class_label=class_label,
            smoothing=self.smoothing,
        )
        # assign and increment the next object id
        self.objects[self.next_object_id] = obj
        self.next_object_id += 1

    def _deregister(self, object_id: int) -> None:
        """
        Deregisters an object from the tracker.
        Args:
            object_id (int): The unique identifier of the object to deregister.
        """
        del self.objects[object_id]

    def _nms_per_class(self, detections: list) -> list:
        """
        Groups detections by class label and applies Non-Maximum Suppression (NMS) to each group.
        #@TODO: improve by region? it really depends on the detector performance

        Args:
            detections (list): A list of detections, where each detection is a tuple (bbox, class_label).

        Returns:
            list: A list of detections after applying NMS, where each detection is a tuple (bbox, class_label).
        """
        grouped = defaultdict(list)
        for box, label in detections:
            grouped[label].append(box)

        results = []
        for label, boxes in grouped.items():
            keep = []
            boxes = np.array(boxes)
            suppressed = [False] * len(boxes)

            for i in range(len(boxes)):
                if suppressed[i]:
                    continue
                box_i = boxes[i]
                keep.append((box_i.tolist(), label))
                for j in range(i + 1, len(boxes)):
                    if suppressed[j]:
                        continue
                    box_j = boxes[j]
                    iou = compute_iou(box_i, box_j)
                    if iou > self.nms_iou_threshold:
                        suppressed[j] = True
            results.extend(keep)
        return results

    def update(self, detections: list) -> dict:
        """
        Updates the tracker with new detections.

        Args:
            detections (list): A list of detections, where each detection is a tuple (bbox, class_label).
            Each bbox is in the format (x1, y1, x2, y2). Class labels are strings (phrases from GDINO).

        Returns:
            dict: A dictionary of tracked objects, where keys are object IDs and values are ObjectTrack instances.
        """
        # Apply NMS
        detections = self._nms_per_class(detections)

        # If no detections, increment disappeared count for all objects
        if len(detections) == 0:
            for object_id in list(self.objects.keys()):
                self.objects[object_id].disappeared += 1
                if self.objects[object_id].disappeared > self.max_disappeared:
                    self._deregister(object_id)
            return self.objects

        # Prepare input centroids, boxes, and labels
        # @TODO: reconcile all box formats across the application
        # e.g., (x1, y1, x2, y2) or (x, y, w, h)
        # or (cx, cy, w, h) or (cx, cy, r)

        input_centroids = []
        input_boxes = []
        input_labels = []

        for bbox, label in detections:
            x1, y1, x2, y2 = bbox
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids.append((cX, cY))
            input_boxes.append(bbox)
            input_labels.append(label)

        # If no objects are being tracked, register all new detections
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self._register(input_centroids[i], input_boxes[i], input_labels[i])
        else:
            # If there are objects being tracked, match them with new detections
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid].centroid for oid in object_ids]
            object_labels = [self.objects[oid].label for oid in object_ids]

            # centroid distance matrix
            D = distance.cdist(np.array(object_centroids), np.array(input_centroids))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            assigned_rows = set()
            assigned_cols = set()

            for row, col in zip(rows, cols):
                if row in assigned_rows or col in assigned_cols:
                    # already assigned
                    continue
                if D[row, col] > self.distance_threshold:
                    # too far apart, skip this assignment
                    continue
                if object_labels[row] != input_labels[col]:
                    # labels do not match, skip this assignment
                    continue

                object_id = object_ids[row]
                self.objects[object_id].update(
                    input_centroids[col], input_boxes[col], input_labels[col]
                )
                assigned_rows.add(row)
                assigned_cols.add(col)

            # Deregister objects that were not assigned
            unused_rows = set(range(0, D.shape[0])) - assigned_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.objects[object_id].disappeared += 1
                if self.objects[object_id].disappeared > self.max_disappeared:
                    self._deregister(object_id)

            # Register new objects that were not assigned
            unused_cols = set(range(0, D.shape[1])) - assigned_cols
            for col in unused_cols:
                self._register(
                    input_centroids[col], input_boxes[col], input_labels[col]
                )

        return self.objects
