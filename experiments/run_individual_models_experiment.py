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
rounds = 50

for model_name in models:
    mc.toggle_ai_model(model_name)

    for _ in tqdm(range(rounds)):
        mc.log_data()
        mc.process_models()

    mc.toggle_ai_model(model_name)
