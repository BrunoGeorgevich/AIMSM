from tqdm import tqdm
from glob import glob
import threading
import traceback
import time
import numpy as np
import sys
import os

os.chdir("..")
sys.path.append(os.getcwd())

from app.Backend.MainController import MainController


KILL_THREAD = False
CURRENT_RUNNING_MODEL = ""

database_path = os.path.join("labs", "single_model_with_AIMSM.csv")

if os.path.exists(database_path):
    os.remove(database_path)

mc = MainController(database_path=database_path, bypass_ros=True)

models = ["Yolo V8", "Fast SAM", "Image Captioning", "Room Classification"]
rounds = 50


def log_data_thread():
    global KILL_THREAD, CURRENT_RUNNING_MODEL
    while not KILL_THREAD:
        mc.log_data()
        mc.register_log([CURRENT_RUNNING_MODEL])
        time.sleep(1 / 300)


try:
    log_thread = threading.Thread(target=log_data_thread)
    log_thread.start()
    time.sleep(1)

    for model_name in models:
        CURRENT_RUNNING_MODEL = ""
        mc.toggle_ai_model(model_name)
        CURRENT_RUNNING_MODEL = model_name

        for _ in tqdm(range(rounds)):
            mc.process_model(model_name)

        CURRENT_RUNNING_MODEL = ""
        mc.toggle_ai_model(model_name)

    time.sleep(1)
except Exception as e:
    print(e)
    traceback.print_exc()

KILL_THREAD = True
