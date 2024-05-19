from tqdm import tqdm
import numpy as np
import threading
import traceback
import time
import sys
import os

os.chdir("..")
sys.path.append(os.getcwd())

from src.AIModules.FastSamModule import FastSamModule
from src.AIModules.YoloV8Module import YoloV8Module
from src.AIModules.ImageCaptioningModule import ImageCaptioningModule
from src.AIModules.RoomClassificationModule import RoomClassificationModule
from app.Backend.MainController import MainController

KILL_THREAD = False
CURRENT_RUNNING_MODEL = ""
FPS_COUNT = 0

def log_data_thread():
    global KILL_THREAD, CURRENT_RUNNING_MODEL, FPS_COUNT
    while not KILL_THREAD:
        mc.log_data(FPS_COUNT)
        mc.register_log([CURRENT_RUNNING_MODEL])
        time.sleep(1 / 300)

database_path = os.path.join("labs", "single_model_without_AIMSM.csv")

if os.path.exists(database_path):
    os.remove(database_path)


mc = MainController(database_path=database_path, do_not_add_models=True, bypass_ros=True)

models = [
    ("Yolo V8", YoloV8Module()),
    ("Fast SAM", FastSamModule()),
    ("Image Captioning", ImageCaptioningModule()),
    ("Room Classification", RoomClassificationModule())
]

input_data = {
    "image": np.zeros((512, 512, 3), dtype=np.uint8),
}

rounds = 50

log_thread = threading.Thread(target=log_data_thread)
log_thread.start()

CURRENT_RUNNING_MODEL = ""
time.sleep(1)

for model in models:
    model[1].initiate()

try:
    for model in models:
        CURRENT_RUNNING_MODEL = model[0]
        for _ in tqdm(range(rounds)):
            it = time.perf_counter_ns()
            model[1].process(input_data)
            et = time.perf_counter_ns()
            elapsed_time = (et - it) / (10**9)
            FPS_COUNT = 1 / elapsed_time if elapsed_time != 0 else 0

except Exception as e:
    print(e)
    traceback.print_exc()

CURRENT_RUNNING_MODEL = ""
time.sleep(1)

KILL_THREAD = True