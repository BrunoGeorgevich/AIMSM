from tqdm import tqdm
import numpy as np
import os
import sys
import threading
import time
import traceback

os.chdir("..")
sys.path.append(os.getcwd())

from src.AIModules.FastSamModule import FastSamModule
from src.AIModules.YoloV8Module import YoloV8Module
from src.AIModules.ImageCaptioningModule import ImageCaptioningModule
from src.AIModules.RoomClassificationModule import RoomClassificationModule
from app.Backend.MainController import MainController

KILL_THREAD = False
FPS_COUNT = 0

def log_data_thread():
    global KILL_THREAD, FPS_COUNT
    while not KILL_THREAD:
        mc.log_data(FPS_COUNT)
        mc.register_log()
        time.sleep(1 / 300)

database_path = os.path.join("labs", "all_models_experiment.csv")

if os.path.exists(database_path):
    os.remove(database_path)

mc = MainController(database_path=database_path, do_not_add_models=True, bypass_ros=True)

models = [
    YoloV8Module(),
    FastSamModule(),
    ImageCaptioningModule(),
    RoomClassificationModule()
]
results = {}
rounds = 50

input_data = {
    "image": np.zeros((512, 512, 3), dtype=np.uint8),
}

try:
    log_thread = threading.Thread(target=log_data_thread)
    log_thread.start()
    time.sleep(1)

    for model in models:
        model.initiate()

    for _ in tqdm(range(rounds)):
        it = time.perf_counter_ns()

        for model in models:
            model.process(input_data)

        et = time.perf_counter_ns()
        elapsed_time = (et - it) / (10**9)
        FPS_COUNT = 1 / elapsed_time if elapsed_time != 0 else 999

    time.sleep(1)
except Exception as e:
    print(e)
    traceback.print_exc()

KILL_THREAD = True