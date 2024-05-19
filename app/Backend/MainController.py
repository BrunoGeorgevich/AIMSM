from app.Backend.DatabaseStrategy.CSVDatabaseStrategy import CSVDatabaseStrategy
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
import psutil
import rospy
import torch
import cv2
import os


@dataclass()
class ResourcesState:
    cpu_usage: float
    ram_usage: float
    gpu_usage: float
    vram_usage: float
    fps_count: float

    def __repr__(self) -> str:
        return f"FPS: {self.fps_count} | CPU: {self.cpu_usage:.2f}% | RAM: {self.ram_usage:.1f}GB | GPU: {self.gpu_usage}% | VRAM: {self.vram_usage:.1f}GB"

    def to_csv(self, sep=";") -> str:
        return f"{self.fps_count}{sep}{self.cpu_usage:.2f}{sep}{self.ram_usage:.1f}{sep}{self.gpu_usage}{sep}{self.vram_usage:.1f}"


class ResourceMonitor:
    def __init__(self, max_size=1):
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

    def __init__(self, database_path="database.csv") -> None:
        super().__init__()

        self.image_provider = ImageProvider(self.__image_provider_handler)
        self.__input_data = {
            "depth": np.zeros((512, 512), dtype=np.uint8),
            "image": np.ones((512, 512, 3), dtype=np.uint8),
        }
        self.__results_data = {}
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

        self.__resources_state = ResourcesState(0, 0, 0, 0, 0)
        self.__resource_monitor = ResourceMonitor()

        self.__aimsm = AIMSM()
        self.__aimsm.add_model("Yolo V8", YoloV8Module())
        self.__aimsm.add_model("Fast SAM", FastSamModule())
        self.__aimsm.add_model("Room Classification", RoomClassificationModule())
        self.__aimsm.add_model("Image Captioning", ImageCaptioningModule())

        self.__cssm = CSSM()
        self.__cssm.add_state("Idle")
        self.__cssm.add_state("Detector")
        self.__cssm.add_state("Segmentor")
        self.__cssm.add_state("All Models")

        self.__cssm.bind("Detector", ["Yolo V8", "Room Classification"])
        self.__cssm.bind("Segmentor", ["Fast SAM", "Room Classification"])
        self.__cssm.bind(
            "All Models",
            # ["Yolo V8", "Fast SAM", "Room Classification"],
            ["Yolo V8", "Fast SAM", "Image Captioning", "Room Classification"],
        )

        self.__cssm.switch("Idle")

        self.__log_controller = LogController(CSVDatabaseStrategy(database_path))
        self.__log_controller.open_database()

    def __del__(self):
        self.__log_controller.close_database()

    @Slot(str)
    def toggle_ai_model(self, ai_model_name: str):
        self.__aimsm.toggle_model(ai_model_name)

    @Slot(str, result=bool)
    def is_ai_model_running(self, ai_model_name: str) -> bool:
        return self.__aimsm.is_model_activated(ai_model_name)


    @Slot()
    def process_models(self):
        i_t = perf_counter()
        results = self.__aimsm.process(self.__input_data)
        e_t = perf_counter()
        elapsed = e_t - i_t
        fps_count = 1 / elapsed if elapsed != 0 else 999

        # if fps_count < 900:
        self.__resources_state.fps_count = fps_count
        self.fpsCounterUpdated.emit()

        # `running_models` is a list that contains the names of the AI models that are currently
        # running and have produced output. It is generated by iterating over the `results`
        # dictionary, which contains the output of each AI model after processing the input data. The
        # names of the models that have produced output are added to the `running_models` list.

        self.__results_data = results

        self.register_log()
        self.__output_data = self.__aimsm.draw_results(self.__input_data, results)

    def set_input_data(self, input_data):
        self.__input_data = input_data

    def register_log(self, running_models=None):
        if running_models is None:
            running_models = self.__aimsm.activated_models()

        if len(running_models) == 1 and running_models[0] == "":
            self.__resources_state.fps_count = 0

        self.__log_controller.write_to_database(
            {"rs": self.__resources_state, "models": running_models}
        )

    @Slot(result=str)
    def get_fps_count(self):
        rs = self.__resource_monitor.get_avg_state()
        rs.fps_count = "999+" if rs.fps_count > 999 else f"{rs.fps_count:.0f}"
        return str(rs)

    @Slot(result=list)
    def get_model_names(self):
        return self.__aimsm.get_model_names()

    @Slot(result=list)
    def get_state_names(self):
        return self.__cssm.get_states()

    @Slot(str)
    def set_state_models(self, state_models):
        self.__aimsm.set_state_models(state_models)

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
    def get_model_result(self, name) -> str:
        if name not in self.__results_data:
            raise ValueError("Model not found")
        return self.__results_data[name]

    @Slot(str, result=str)
    def get_model_output_type(self, name):
        return self.__aimsm.get_model_output_type(name)

    @Slot()
    def log_data(self):
        rs = self.__resources_state

        rs.cpu_usage = psutil.cpu_percent()
        rs.ram_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        rs.gpu_usage = torch.cuda.utilization()
        rs.vram_usage = torch.cuda.memory_allocated() / 1024**3

        self.__resource_monitor.add_state(rs)
        self.__resources_state = self.__resource_monitor.get_avg_state()

        # if isinstance(self.fps_counter, float):
        #     if self.fps_counter > 999:
        #         fps_str = "-"
        #     else:
        #         fps_str = f"{self.fps_counter:.2f}"
        # self.__log_controller.log_data()

    def __image_provider_handler(self, path: str, size: int, requestedSize: int):
        method = path.split("/")[0]
        if method == "none":
            return toQImage(np.zeros((480, 640, 3), dtype=np.uint8))
        elif method == "image":
            return toQImage(self.__input_data["image"])
        elif method == "depth":
            return toQImage(self.__input_data["depth"])
        elif method == "Yolo V8":
            if self.__aimsm.is_model_activated("Yolo V8"):
                return toQImage(self.__output_data["Yolo V8"])
            else:
                return toQImage(np.zeros((480, 640, 3), dtype=np.uint8))
        elif method == "Fast SAM":
            if self.__aimsm.is_model_activated("Fast SAM"):
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
