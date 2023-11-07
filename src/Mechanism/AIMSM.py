from src.AIModules.AIModule import AIModule
import numpy as np


class AIMSM:
    def __init__(self) -> None:
        self.__models = {}

    def add_model(self, name: str, model: AIModule) -> None:
        if model.is_initialized():
            raise ValueError("Model is already initialized")
        self.__models[name] = model

    def initiate_model(self, name):
        if name not in self.__models:
            raise ValueError("Model not found")
        self.__models[name].initiate()

    def deinitiate_model(self, name):
        if name not in self.__models:
            raise ValueError("Model not found")
        self.__models[name].deinitiate()

    def is_model_initialized(self, name):
        if name not in self.__models:
            raise ValueError("Model not found")
        return self.__models[name].is_initialized()

    def toggle_model(self, name):
        if name not in self.__models:
            raise ValueError("Model not found")
        if self.__models[name].is_initialized():
            self.__models[name].deinitiate()
        else:
            self.__models[name].initiate()

    def get_model_names(self):
        return list(self.__models.keys())

    def get_model_output_type(self, name):
        if name not in self.__models:
            raise ValueError("Model not found")
        return self.__models[name].get_output_type()

    def process(self, input_data: dict) -> dict:
        results = {}

        for name, model in self.__models.items():
            if model.is_initialized():
                results[name] = model.process(input_data)
            else:
                results[name] = np.zeros((480, 640, 3), dtype=np.uint8)

        return results

    def draw_results(self, input_data, processed_results):
        results = {}
        for name, model in self.__models.items():
            processed_result = processed_results[name]
            if processed_result is None:
                results[name] = np.zeros((430, 640, 3), dtype=np.uint8)
            if model.is_initialized():
                results[name] = model.draw_results(input_data, processed_result)
        return results
