import numpy as np
import ultralytics.engine.results
from ultralytics import YOLO
from typing import List


class YOLOModel:
    def __init__(self):
        """
        Class constructor
        """
        self.model = YOLO("yolov8n.pt")

    def predict(self, frame: np.ndarray, conf: float, classes: int or List[int], iou: float) ->\
            ultralytics.engine.results.Results:
        """
        Detects object on frame, some parameters have been added.
        More can be found here: https://docs.ultralytics.com/modes/predict/#inference-arguments
        :param frame: Image frame where objects will be detected
        :param conf: Confidence threshold (float from 0 to 1)
        :param classes: Classes to detect (Can be a single int or list of ints if multiple)
        :param iou: intersection over union, overlap threshold to filter objects on top of each other
        :return: Ultralytics results object with prediction
        """
        return self.model.predict(frame, conf=conf, classes=classes, iou=iou)
