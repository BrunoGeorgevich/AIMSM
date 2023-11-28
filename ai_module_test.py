from src.AIModules.YoloV8Module import YoloV8Module
from src.AIModules.RoomClassificationModule import RoomClassificationModule
from src.AIModules.FastSamModule import FastSamModule
from src.AIModules.SamModule import SamModule

from app.Backend.MainController import MainController

from tqdm import tqdm
from glob import glob
import numpy as np
import cv2
import os

num_databases = len(glob(os.path.join("database*.csv")))
database_path = f"database{'' if num_databases == 0 else num_databases}.csv"

mc = MainController(database_path)
# mc.toggle_ai_model("Yolo V8")
# mc.toggle_ai_model("Fast SAM")
mc.toggle_ai_model("Room Classification")
# mc.toggle_ai_model("Image Captioning")

image = cv2.imread("bus.jpg")
image = cv2.resize(image, (640, 480))
mc.set_input_data({"image": image})

for i in tqdm(range(300)):
    mc.log_data()
    mc.process_models()
