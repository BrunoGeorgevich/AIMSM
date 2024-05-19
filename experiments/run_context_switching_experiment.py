from time import perf_counter
from enum import Enum
from glob import glob
from tqdm import tqdm
import numpy as np
import signal
import sys
import os

os.chdir("..")
sys.path.append(os.getcwd())

signal.signal(signal.SIGINT, signal.SIG_DFL)

from app.Backend.MainController import MainController
import threading
import traceback
import time

KILL_THREAD = False
CURRENT_RUNNING_MODEL = ""

database_path = os.path.join("labs", "context_switching_experiment_database.csv")

if os.path.exists(database_path):
    os.remove(database_path)

mc = MainController(database_path=database_path)
script = open("script.txt", "r").read().split("\n")

def log_data_thread():
    global KILL_THREAD, CURRENT_RUNNING_MODEL
    while not KILL_THREAD:
        mc.log_data()
        mc.register_log([CURRENT_RUNNING_MODEL])
        time.sleep(1 / 300)


log_thread = threading.Thread(target=log_data_thread)
log_thread.start()

CURRENT_RUNNING_MODEL = ""
time.sleep(1)

mc.toggle_ai_model("Room Classification")
CURRENT_RUNNING_MODEL = "Room Classification"
last_model = ""
try:
    for model_name in tqdm(script):
        if last_model != model_name:
            CURRENT_RUNNING_MODEL = ""
            if model_name != "":
                mc.toggle_ai_model(model_name)
            if last_model != "":
                mc.toggle_ai_model(last_model)

        if model_name == "":
            CURRENT_RUNNING_MODEL = "Room Classification"
        else:
            CURRENT_RUNNING_MODEL = f"{model_name} + Room Classification"

        mc.process_models()
        last_model = model_name
except Exception as e:
    print(e)
    traceback.print_exc()

CURRENT_RUNNING_MODEL = ""
time.sleep(1)

KILL_THREAD = True
