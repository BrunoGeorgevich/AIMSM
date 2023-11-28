from time import perf_counter
from tqdm import tqdm
from glob import glob
import numpy as np
import sys
import os

os.chdir("..")
sys.path.append(os.getcwd())

from app.Backend.MainController import MainController

num_databases = len(glob(os.path.join("database*.csv")))
database_path = f"database{'' if num_databases == 0 else num_databases}.csv"

mc = MainController(database_path)

models = ["Yolo V8", "Fast SAM", "Image Captioning", "Room Classification"]
model_idx = 0

print("Starting experiment...")

i_t = perf_counter()
time_window = 10

mc.toggle_ai_model(models[model_idx])
while True:
    e_t = perf_counter()
    mc.log_data()
    mc.process_models()
    if e_t - i_t >= time_window:
        elapsed = e_t - i_t
        i_t = e_t
        model_idx += 1

        if model_idx == len(models):
            break

        mc.toggle_ai_model(models[model_idx - 1])
        mc.toggle_ai_model(models[model_idx])
        print(f"Changed model in {elapsed:.3f} s to {models[model_idx]}")


# for model_name in models:
#     mc.toggle_ai_model(model_name)

#     for _ in tqdm(range(rounds)):
#         mc.log_data()
#         mc.process_models()

#     mc.toggle_ai_model(model_name)
