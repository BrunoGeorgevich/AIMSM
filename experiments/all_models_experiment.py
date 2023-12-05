from time import perf_counter_ns
from tqdm import tqdm
import numpy as np
import sys
import os

os.chdir("..")
sys.path.append(os.getcwd())

from app.Backend.MainController import MainController

mc = MainController()

input_data = {
    "image": np.zeros((512, 512, 3), dtype=np.uint8),
}

models = ["Yolo V8", "Fast SAM", "Image Captioning", "Room Classification"]
results = {}
rounds = 250

for model_name in models:
    mc.toggle_ai_model(model_name)

for _ in tqdm(range(rounds)):
    mc.log_data()
    mc.register_log()

    mc.process_models()

    mc.log_data()
    mc.register_log()
