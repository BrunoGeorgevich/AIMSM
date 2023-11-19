from unittest import result
from src.AIModules.ImageCaptioningModule import ImageCaptioningModule
from src.AIModules.RoomClassificationModule import RoomClassificationModule
from src.AIModules.FastSamModule import FastSamModule
from src.AIModules.YoloV8Module import YoloV8Module
from app.Backend.LogController import LogController
from app.Backend.Base import ImageProvider
from src.Mechanism.AIMSM import AIMSM
from src.Mechanism.CSSM import CSSM

from PySide2.QtCore import QObject, Slot, Signal
from sensor_msgs.msg import CompressedImage
from dataclasses import dataclass
from PySide2.QtGui import QImage
from collections import deque
from time import perf_counter
import numpy as np
import pynvml
import psutil
import rospy
import cv2


@dataclass()
class ResourcesState:
    cpu_usage: float
    ram_usage: float
    gpu_usage: float
    vram_usage: float
    fps_count: float


class ResourceMonitor:
    def __init__(self, max_size=10):
        self.__state_queue = deque(maxlen=max_size)

    def add_state(self, state: ResourcesState):
        self.__state_queue.append(state)

    def get_avg_state(self):
        cpu_usage = []
        ram_usage = []
        gpu_usage = []
        vram_usage = []
        fps_count = []

        for state in self.__state_queue:
            cpu_usage.append(state.cpu_usage)
            ram_usage.append(state.ram_usage)
            gpu_usage.append(state.gpu_usage)
            vram_usage.append(state.vram_usage)
            fps_count.append(state.fps_count)

        cpu_usage_avg = np.mean(cpu_usage)
        ram_usage_avg = np.mean(ram_usage)
        gpu_usage_avg = np.mean(gpu_usage)
        vram_usage_avg = np.mean(vram_usage)
        fps_count_avg = np.mean(fps_count)

        return ResourcesState(
            cpu_usage_avg, ram_usage_avg, gpu_usage_avg, vram_usage_avg, fps_count_avg
        )


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
    stateSwitched = Signal()

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

        self.__resources_state = ResourcesState(-1, -1, -1, -1, -1)
        self.__resource_monitor = ResourceMonitor()

        self.__aimsm = AIMSM()
        self.__aimsm.add_model("Yolo V8", YoloV8Module())
        self.__aimsm.add_model("Fast SAM", FastSamModule())
        self.__aimsm.add_model("Image Captioning", ImageCaptioningModule())
        self.__aimsm.add_model("Room Classification", RoomClassificationModule())

        self.__cssm = CSSM()
        self.__cssm.add_state("Idle")
        self.__cssm.add_state("Detector")
        self.__cssm.add_state("Segmentor")
        self.__cssm.add_state("All Models")

        self.__cssm.bind("Detector", ["Yolo V8", "Room Classification"])
        self.__cssm.bind("Segmentor", ["Fast SAM", "Room Classification"])
        self.__cssm.bind(
            "All Models",
            ["Yolo V8", "Fast SAM", "Image Captioning", "Room Classification"],
        )

        self.__cssm.switch("Idle")

        self.__log_controller = LogController()

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
        fps_count = 1 / elapsed if elapsed != 0 else 999

        if fps_count < 900:
            self.__resources_state.fps_count = fps_count
            self.fpsCounterUpdated.emit()

        self.__output_data = self.__aimsm.draw_results(self.__input_data, results)

    @Slot(result=str)
    def get_fps_count(self):
        rs = self.__resource_monitor.get_avg_state()
        rs.fps_count = "999+" if rs.fps_count > 999 else f"{rs.fps_count:.0f}"
        return f"FPS: {rs.fps_count} | CPU: {rs.cpu_usage:.2f}% | RAM: {rs.ram_usage:.1f}GB | GPU: {rs.gpu_usage}% | VRAM: {rs.vram_usage:.1f}GB"

    @Slot(result=list)
    def get_model_names(self):
        return self.__aimsm.get_model_names()

    @Slot(result=list)
    def get_state_names(self):
        return self.__cssm.get_states()

    @Slot(str)
    def switch_state(self, state_name):
        self.__cssm.switch(state_name)
        self.__aimsm.set_state_models(self.__cssm.state_models())
        self.stateSwitched.emit()

    @Slot(result=str)
    def get_current_state(self):
        return self.__cssm.current_state()

    @Slot(str, result=str)
    def get_model_output(self, name) -> str:
        if name not in self.__output_data:
            raise ValueError("Model not found")
        return self.__output_data[name]

    @Slot(str, result=str)
    def get_model_output_type(self, name):
        return self.__aimsm.get_model_output_type(name)

    @Slot()
    def log_data(self):
        rs = self.__resources_state
        rs.cpu_usage = psutil.cpu_percent()
        rs.ram_usage = psutil.virtual_memory().used / 1024**3

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        rs.gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        rs.vram_usage = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3
        pynvml.nvmlShutdown()

        self.__resource_monitor.add_state(rs)
        self.__resources_state = self.__resource_monitor.get_avg_state()

        # if isinstance(self.fps_counter, float):
        #     if self.fps_counter > 999:
        #         fps_str = "-"
        #     else:
        #         fps_str = f"{self.fps_counter:.2f}"
        # self.__log_controller.log_data()
        pass

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
