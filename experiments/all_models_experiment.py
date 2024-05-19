from time import perf_counter_ns
from tqdm import tqdm
import traceback
import numpy as np
import sys
import os

os.chdir("..")
sys.path.append(os.getcwd())

from app.Backend.MainController import MainController
import threading
import time

KILL_THREAD = False

database_path = os.path.join("labs", "all_model_experiment.csv")

if os.path.exists(database_path):
    os.remove(database_path)

mc = MainController(database_path=database_path)

models = ["Yolo V8", "Fast SAM", "Image Captioning", "Room Classification"]
results = {}
rounds = 250


def log_data_thread():
    global KILL_THREAD
    while not KILL_THREAD:
        mc.log_data()
        mc.register_log()
        time.sleep(1 / 300)

try:
    log_thread = threading.Thread(target=log_data_thread)
    log_thread.start()
    time.sleep(1)

    for model_name in models:
        mc.toggle_ai_model(model_name)

    for _ in tqdm(range(rounds)):
        mc.process_models()

    time.sleep(1)
except Exception as e:
    print(e)
    traceback.print_exc()

KILL_THREAD = True