from time import perf_counter
from src.AIModules.ImageCaptioningModule import ImageCaptioningModule
from src.AIModules.RoomClassificationModule import RoomClassificationModule
from src.AIModules.FastSamModule import FastSamModule
from src.AIModules.YoloV8Module import YoloV8Module
from app.Backend.Base import ImageProvider
from src.Mechanism.AIMSM import AIMSM

from PySide2.QtCore import QObject, Slot, Signal
from sensor_msgs.msg import CompressedImage
from PySide2.QtGui import QImage
import numpy as np
import pynvml
import psutil
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
    fpsCounterUpdated = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.image_provider = ImageProvider(self.__image_provider_handler)
        self.__input_data = {
            "depth": np.zeros((480, 640), dtype=np.uint8),
            "image": np.ones((480, 640, 3), dtype=np.uint8),
        }
        self.__output_data = {}
        self.fps_counter = "-"

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
        self.__aimsm.add_model("Yolo V8", YoloV8Module())
        self.__aimsm.add_model("Fast SAM", FastSamModule())
        self.__aimsm.add_model("Image Captioning", ImageCaptioningModule())
        self.__aimsm.add_model("Room Classification", RoomClassificationModule())

    @Slot(str)
    def toggle_ai_model(self, ai_model_name: str):
        self.__aimsm.toggle_model(ai_model_name)

    @Slot(str, result=bool)
    def is_ai_model_running(self, ai_model_name: str) -> bool:
        return self.__aimsm.is_model_initialized(ai_model_name)

    @Slot()
    def process_models(self):
        i_t = perf_counter()
        results = self.__aimsm.process(self.__input_data)
        e_t = perf_counter()
        elapsed = e_t - i_t
        self.fps_counter = 1 / elapsed if elapsed != 0 else 999
        self.fpsCounterUpdated.emit()
        self.__output_data = self.__aimsm.draw_results(self.__input_data, results)

    @Slot(result=str)
    def get_fps_count(self):
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        fps_str = self.fps_counter

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        vram_usage = (
            pynvml.nvmlDeviceGetMemoryInfo(handle).used
            / pynvml.nvmlDeviceGetMemoryInfo(handle).total
        )
        pynvml.nvmlShutdown()

        if isinstance(self.fps_counter, float):
            if self.fps_counter > 999:
                fps_str = "999+"
            else:
                fps_str = f"{self.fps_counter:.2f}"

        return f"FPS: {fps_str} | CPU: {cpu_usage:.2f}% | RAM: {ram_usage:.2f}% | GPU: {gpu_usage}% | VRAM: {vram_usage*100:.2f}%"

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
        elif method == "Yolo V8":
            if self.__aimsm.is_model_initialized("Yolo V8"):
                return toQImage(self.__output_data["Yolo V8"])
            else:
                return toQImage(np.zeros((480, 640, 3), dtype=np.uint8))
        elif method == "Fast SAM":
            if self.__aimsm.is_model_initialized("Fast SAM"):
                return toQImage(self.__output_data["Fast SAM"])
            else:
                return toQImage(np.zeros((480, 640, 3), dtype=np.uint8))
        elif method == "Image Captioning":
            return toQImage(np.zeros((480, 640, 3), dtype=np.uint8))
        elif method == "Room Classification":
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
