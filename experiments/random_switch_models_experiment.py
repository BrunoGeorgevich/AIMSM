from time import perf_counter_ns
from random import choice
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
rounds = 250

for _ in tqdm(range(rounds)):
    model_name = choice(models)

    times = {"initiate": [], "deinitiate": []}
    mc.log_data()
    mc.register_log()

    i_t = perf_counter_ns()
    mc.toggle_ai_model(model_name)
    e_t = perf_counter_ns()
    initiate_time = (e_t - i_t) / (10**9)
    times["initiate"].append(initiate_time)

    mc.log_data()
    mc.register_log()

    mc.process_models()

    mc.log_data()
    mc.register_log()

    i_t = perf_counter_ns()
    mc.toggle_ai_model(model_name)
    e_t = perf_counter_ns()
    deinitiate_time = (e_t - i_t) / (10**9)
    times["deinitiate"].append(deinitiate_time)

initiate_time_mean = np.mean(times["initiate"])
deinitiate_time_mean = np.mean(times["deinitiate"])

initiate_time_std = np.std(times["initiate"])
deinitiate_time_std = np.std(times["deinitiate"])

print("-" * 50)
print("Random model switching experiment")
print(f"Mean initiate time: {initiate_time_mean:.3f} ({initiate_time_std:.3f}) s")
print(f"Mean deinitiate time: {deinitiate_time_mean:.3f} ({deinitiate_time_std:.3f}) s")
print("-" * 50)
