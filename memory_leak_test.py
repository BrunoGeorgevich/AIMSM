from src.AIModules.ImageCaptioningModule import ImageCaptioningModule
from src.AIModules.YoloV8Module import YoloV8Module
from src.AIModules.RoomClassificationModule import RoomClassificationModule
from src.AIModules.FastSamModule import FastSamModule
from src.AIModules.SamModule import SamModule

import numpy as np
import psutil
import torch
import os
import gc


import ctypes.util
import ctypes

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libc.malloc_trim(ctypes.c_int(0))

os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_SCHEDULE"] = "STATIC"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"


def print_ram(step):
    print(
        f"#{step} RAM: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB | VRAM: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
    )


def run():
    module1 = RoomClassificationModule()
    module2 = YoloV8Module()
    module3 = FastSamModule()
    module4 = ImageCaptioningModule()

    module1.initiate()
    module2.initiate()
    module3.initiate()
    module4.initiate()

    input_data = {"image": np.zeros((512, 512, 3), dtype=np.uint8)}

    for i in range(100):
        output = module1.process(input_data)
        output = module1.draw_results(input_data, output)
        output = module2.process(input_data)
        output = module2.draw_results(input_data, output)
        output = module3.process(input_data)
        output = module3.draw_results(input_data, output)
        output = module4.process(input_data)
        output = module4.draw_results(input_data, output)
        print_ram(i)


run()
