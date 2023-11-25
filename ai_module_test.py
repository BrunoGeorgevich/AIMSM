from src.AIModules.YoloV8Module import YoloV8Module
from src.AIModules.RoomClassificationModule import RoomClassificationModule
from src.AIModules.FastSamModule import FastSamModule
from src.AIModules.SamModule import SamModule

import numpy as np
import cv2

module = YoloV8Module()
module.initiate()

image = cv2.imread("bus.jpg")

for i in range(300):
    input_data = {"image": image}
    output = module.process(input_data)
    module.draw_results(input_data, output)
