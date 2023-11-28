from time import perf_counter
from enum import Enum
from glob import glob
import numpy as np
import signal
import sys
import os

os.chdir("..")
sys.path.append(os.getcwd())

signal.signal(signal.SIGINT, signal.SIG_DFL)

from app.Backend.MainController import MainController


class RoomState(Enum):
    LIVING_ROOM = "living room"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    BEDROOM = "bedroom"
    DINING_ROOM = "dining room"
    NONE = "none"


class State:
    def __init__(self, thresh=5):
        self.__current_room = RoomState.NONE
        self.__iterations = 0
        self.__reset_count = 3
        self.__thresh = thresh

    def update(self, room_state):
        if self.__current_room != room_state:
            self.__reset_count -= 1
            if self.__reset_count <= 0:
                self.__current_room = room_state
                self.__iterations = 0
                self.__reset_count = 3
        else:
            print(self.__current_room, self.__iterations)
            self.__iterations += 1
            self.__reset_count = 5

    def get_current_room(self):
        if self.__iterations <= self.__thresh:
            return RoomState.NONE
        return self.__current_room


num_databases = len(glob(os.path.join("database*.csv")))
database_path = f"database{'' if num_databases == 0 else num_databases}.csv"

mc = MainController(database_path)
mc.toggle_ai_model("Room Classification")

state = State(10)

while True:
    mc.log_data()
    mc.process_models()
    res = mc.get_model_result("Room Classification")
    out = mc.get_model_output("Room Classification")

    print(res, state.get_current_room())
    state.update(RoomState(res["label"]))

    if state.get_current_room() == RoomState.BEDROOM:
        mc.set_state_models(["Image Captioning", "Room Classification"])
    elif state.get_current_room() == RoomState.BATHROOM:
        mc.set_state_models(["Yolo V8", "Room Classification"])
    elif state.get_current_room() == RoomState.KITCHEN:
        mc.set_state_models(["Fast SAM", "Room Classification"])
    elif state.get_current_room() == RoomState.NONE:
        mc.set_state_models(["Room Classification"])
