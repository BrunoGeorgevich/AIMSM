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
rounds = 50

for model_name in models:
    if model_name not in results:
        results[model_name] = {"initiate": [], "deinitiate": []}

    for _ in tqdm(range(rounds)):
        i_t = perf_counter_ns()
        mc.toggle_ai_model(model_name)
        e_t = perf_counter_ns()
        initiate_time_mean = (e_t - i_t) / (10**9)
        results[model_name]["initiate"].append(initiate_time_mean)

        mc.log_data()
        mc.register_log()

        mc.process_models()

        mc.log_data()
        mc.register_log()

        i_t = perf_counter_ns()
        mc.toggle_ai_model(model_name)
        e_t = perf_counter_ns()
        deinitiate_time_mean = (e_t - i_t) / (10**9)
        results[model_name]["deinitiate"].append(deinitiate_time_mean)

for model_name in results:
    initiate_time_mean = np.mean(results[model_name]["initiate"])
    deinitiate_time_mean = np.mean(results[model_name]["deinitiate"])

    initiate_time_std = np.std(results[model_name]["initiate"])
    deinitiate_time_std = np.std(results[model_name]["deinitiate"])

    print("-" * 50)
    print(f"Model: {model_name}")
    print(f"Mean initiate time: {initiate_time_mean:.3f} ({initiate_time_std:.3f}) s")
    print(
        f"Mean deinitiate time: {deinitiate_time_mean:.3f} ({deinitiate_time_std:.3f}) s"
    )

print("-" * 50)
