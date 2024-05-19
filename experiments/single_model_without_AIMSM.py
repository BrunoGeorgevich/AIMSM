from time import perf_counter_ns
from tqdm import tqdm
import numpy as np
import traceback
import sys
import os

os.chdir("..")
sys.path.append(os.getcwd())

from app.Backend.MainController import MainController
import threading
import time

KILL_THREAD = False
CURRENT_RUNNING_MODEL = ""

database_path = os.path.join("labs", "single_model_without_AIMSM.csv")

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

log_thread = threading.Thread(target=log_data_thread)
log_thread.start()

CURRENT_RUNNING_MODEL = ""
time.sleep(1)

for model_name in models:
    mc.toggle_ai_model(model_name)

try:
    for model_name in models:
        CURRENT_RUNNING_MODEL = model_name
        for _ in tqdm(range(rounds)):
            mc.process_model(model_name)
except Exception as e:
    print(e)
    traceback.print_exc()

CURRENT_RUNNING_MODEL = ""
time.sleep(1)

KILL_THREAD = True