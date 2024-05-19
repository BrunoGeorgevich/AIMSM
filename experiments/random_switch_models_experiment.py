from time import perf_counter_ns
from random import choice
from tqdm import tqdm
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

database_path = os.path.join("labs", "random_switch_models_experiment.csv")

if os.path.exists(database_path):
    os.remove(database_path)

mc = MainController(database_path=database_path)

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

    last_model = None

    for _ in tqdm(range(rounds)):
        if last_model is None:
            model_name = choice(models)
        else:
            model_name = choice([model for model in models if model != last_model])

        mc.toggle_ai_model(model_name)
        CURRENT_RUNNING_MODEL = model_name
        for _ in range(5):
            mc.process_model(model_name)
        mc.toggle_ai_model(model_name)
        CURRENT_RUNNING_MODEL = ""

        last_model = model_name


    time.sleep(1)
except Exception as e:
    print(e)
    traceback.print_exc()

KILL_THREAD = True
