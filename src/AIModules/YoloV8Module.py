from src.AIModules.AIModule import AIModule

from ultralytics import YOLO
from random import randint
import numpy as np
import cv2


class YoloV8Module(AIModule):
    """This class is the implementation of the model Yolo V8 as an AIModule"""

    __model = None

    def __init__(self) -> None:
        """
        The function initializes an empty dictionary called "__colors" that will hold the colors for each class
        """
        self.__colors = {}

    def initiate(self, model_path: str = "yolov8n.pt") -> None:
        """
        The `initiate` function initializes a YOLO object with a specified model path.

        :param model_path: The `model_path` parameter is a string that represents the file path to the
        YOLOv8 model file. This file contains the pre-trained weights and architecture of the YOLOv8
        model, defaults to yolov8n.pt
        :type model_path: str (optional)
        """
        self.__model = YOLO(model_path)

    def deinitiate(self) -> None:
        """
        The function deinitiate sets the value of the __model attribute to None.
        """
        self.__model = None

    def process(self, image: np.ndarray) -> list:
        """
        The function processes an image using a pre-trained model and returns the predicted results.

        :param image: The image parameter is an input image that you want to process. It should be a
        numpy array representing the image
        :type image: np.ndarray
        :return: a list of results.
        """
        if self.__model is None:
            raise ValueError("Model is not initiated")

        results = self.__model.predict(image, conf=0.6, iou=0.4)
        return results

    def draw_results(self, image: np.ndarray, results: list) -> np.ndarray:
        """
        The function takes an image and a list of results, and draws bounding boxes and labels on the
        image based on the results.

        :param image: The `image` parameter is a NumPy array representing an image
        :type image: np.ndarray
        :param results: The `results` parameter is a list of objects that contain information about the
        detected objects in an image. Each object in the list has the following attributes:
        :type results: list
        :return: a processed image with bounding boxes and class labels drawn on it.
        """
        processed_image = image.copy()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                c = int(box.cls)
                c_name = self.__model.names[c]

                if c not in self.__colors:
                    self.__colors[c] = (
                        randint(0, 255),
                        randint(0, 255),
                        randint(0, 255),
                    )

                cv2.rectangle(
                    processed_image,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    self.__colors[c],
                    2,
                )

                cv2.putText(
                    processed_image,
                    c_name,
                    (int(x1), int(y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    self.__colors[c],
                    2,
                )

        return processed_image
