from src.AIModules.ImageCaptioningModule import ImageCaptioningModule
from src.AIModules.FastSamModule import FastSamModule
from src.AIModules.YoloV8Module import YoloV8Module
from app.Backend.Base import ImageProvider
from src.Mechanism.AIMSM import AIMSM

from sensor_msgs.msg import CompressedImage
from PySide2.QtCore import QObject, Slot
from PySide2.QtGui import QImage
import numpy as np
import rospy
import cv2


def toQImage(image: np.ndarray) -> QImage:
    """Function that generates an image to be used by
    Qt frontend from an opencv or numpy image array.

    :param image: Input image.
    :type image: np.ndarray
    :return: Qt image from input image.
    :rtype: QImage
    """
    if not (np.max(image) <= 255):
        raise ValueError("Image values outside higher limit (0-255)")
    if not (np.min(image) >= 0):
        raise ValueError("Image values outside lower limit (0-255)")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    height = image.shape[0]
    width = image.shape[1]
    bytes_per_line = 3 * width
    qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
    return qimage


class MainController(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.image_provider = ImageProvider(self.__image_provider_handler)
        self.__input_data = {
            "depth": np.zeros((480, 640), dtype=np.uint8),
            "image": np.ones((480, 640, 3), dtype=np.uint8),
        }
        self.__output_data = {}

        rospy.init_node("MainControllerNode")
        rospy.Subscriber(
            "/RobotAtVirtualHome/VirtualCameraRGB",
            CompressedImage,
            self.__rgb_callback,
        )
        rospy.Subscriber(
            "/RobotAtVirtualHome/VirtualCameraDepth",
            CompressedImage,
            self.__depth_callback,
        )

        self.__aimsm = AIMSM()
        self.__aimsm.add_model("YoloV8", YoloV8Module())
        self.__aimsm.add_model("FastSAM", FastSamModule())
        self.__aimsm.add_model("ImageCaptioning", ImageCaptioningModule())

    @Slot(str)
    def toggle_ai_model(self, ai_model_name: str):
        self.__aimsm.toggle_model(ai_model_name)

    @Slot(str, result=bool)
    def is_ai_model_running(self, ai_model_name: str) -> bool:
        return self.__aimsm.is_model_initialized(ai_model_name)

    @Slot()
    def process_models(self):
        results = self.__aimsm.process(self.__input_data)
        self.__output_data = self.__aimsm.draw_results(self.__input_data, results)

    @Slot(result=list)
    def get_model_names(self):
        return self.__aimsm.get_model_names()

    @Slot(str, result=str)
    def get_model_output(self, name) -> str:
        if name not in self.__output_data:
            raise ValueError("Model not found")
        return self.__output_data[name]

    @Slot(str, result=str)
    def get_model_output_type(self, name):
        return self.__aimsm.get_model_output_type(name)

    def __image_provider_handler(self, path: str, size: int, requestedSize: int):
        method = path.split("/")[0]
        if method == "none":
            return toQImage(np.zeros((480, 640, 3), dtype=np.uint8))
        elif method == "image":
            return toQImage(self.__input_data["image"])
        elif method == "depth":
            return toQImage(self.__input_data["depth"])
        elif method == "YoloV8":
            if self.__aimsm.is_model_initialized("YoloV8"):
                return toQImage(self.__output_data["YoloV8"])
            else:
                return toQImage(np.zeros((480, 640, 3), dtype=np.uint8))
        elif method == "FastSAM":
            if self.__aimsm.is_model_initialized("FastSAM"):
                return toQImage(self.__output_data["FastSAM"])
            else:
                return toQImage(np.zeros((480, 640, 3), dtype=np.uint8))
        elif method == "ImageCaptioning":
            return toQImage(np.zeros((480, 640, 3), dtype=np.uint8))
        else:
            raise ValueError("Invalid path")

    def __rgb_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.__input_data["image"] = img.copy()

    def __depth_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.__input_data["depth"] = img.copy()
