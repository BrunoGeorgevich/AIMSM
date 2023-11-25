from src.AIModules.AIModule import AIModule, ModuleOutput

from random import randint
import numpy as np
import torch
import cv2
import gc
import os

os.environ["YOLO_VERBOSE"] = "false"

from ultralytics import YOLO


class YoloV8Module(AIModule):
    """This class is the implementation of the model Yolo V8 as an AIModule"""

    __model = None
    __initialized = False

    def __init__(self) -> None:
        """
        The function initializes an empty dictionary called "__colors" that will hold the colors for each class
        """
        self.__colors = {}

    @torch.no_grad()
    def initiate(self, model_path: str = "YoloV8.pt") -> None:
        """
        The `initiate` function initializes a YOLO object with a specified model path.

        :param model_path: The `model_path` parameter is a string that represents the file path to the
        YOLOv8 model file. This file contains the pre-trained weights and architecture of the YOLOv8
        model, defaults to yolov8n.pt
        :type model_path: str (optional)
        """
        self.__model = YOLO(model_path)
        self.__initialized = True

    @torch.no_grad()
    def deinitiate(self) -> None:
        """
        The function deinitiate sets the value of the __model attribute to None.
        """
        del self.__model
        self.__model = None
        self.__initialized = False

        try:
            torch._C._cuda_clearCublasWorkspaces()
            torch._dynamo.reset()
        except AttributeError:
            pass

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def process(self, input_data: dict) -> list:
        """
        The function processes an image using a pre-trained model and returns the predicted results.

        :param input_data: Input data dictionary, which must contains the image to be processed.
        {
            "image": np.ndarray
            ...
        }
        :type input_data: dict
        :return: a list of results.
        """
        with torch.no_grad():
            image = input_data.get("image", None)

            if image is None:
                raise ValueError("Image is not provided")

            if self.__model is None:
                raise ValueError("Model is not initiated")

            results = self.__model.predict(image, conf=0.4, iou=0.4)

            del image
            gc.collect()
            torch.cuda.empty_cache()

            return results

    @torch.no_grad()
    def draw_results(self, input_data: dict, results: list) -> np.ndarray:
        """
        The function takes an image and a list of results, and draws bounding boxes and labels on the
        image based on the results.

        :param input_data: Input data dictionary, which must contains the image to be processed.
        {
            "image": np.ndarray
            ...
        }
        :param results: The `results` parameter is a list of objects that contain information about the
        detected objects in an image. Each object in the list has the following attributes:
        :type results: list
        :return: a processed image with bounding boxes and class labels drawn on it.
        """
        image = input_data.get("image", None)

        if image is None:
            raise ValueError("Image is not provided")

        processed_image = image.copy()

        if results is None:
            return None

        try:
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
        except AttributeError:
            return None

        del results
        del input_data
        gc.collect()
        torch.cuda.empty_cache()

        return processed_image

    def is_initialized(self) -> bool:
        """
        The function returns the value of the __initialized attribute.

        :return: a boolean value indicating whether the model is initialized or not.
        :rtype: bool
        """
        return self.__initialized

    def get_output_type(self) -> str:
        """
        The function returns the type of the output of the module.

        :return: a string indicating the type of the output of the module.
        :rtype: str
        """

        return ModuleOutput.IMAGE.name
