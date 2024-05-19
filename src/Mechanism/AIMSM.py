from src.AIModules.AIModule import AIModule
from threading import Thread
import numpy as np


class AIMSM:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__models = {}

    def add_model(self, name: str, model: AIModule) -> None:
        if model.is_initialized():
            raise ValueError("Model is already initialized")
        self.__models[name] = model

    def activate_model(self, name):
        if name not in self.__models:
            raise ValueError("Model not found")
        self.__models[name].initiate()

    def deactivate_model(self, name):
        if name not in self.__models:
            raise ValueError("Model not found")
        self.__models[name].deinitiate()

    def is_model_activated(self, name):
        if name not in self.__models:
            raise ValueError("Model not found")
        return self.__models[name].is_initialized()

    def activated_models(self):
        return [name for name, model in self.__models.items() if model.is_initialized()]

    def toggle_model(self, name: str | list):
        def toggle_model_thread(name):
            if name not in self.__models:
                raise ValueError("Model not found")
            if self.__models[name].is_initialized():
                self.disable_model(name)
            else:
                self.enable_model(name)

        if isinstance(name, str):
            toggle_model_thread(name)
        elif isinstance(name, list):
            for n in name:
                toggle_model_thread(n)
                # t = Thread(target=toggle_model_thread, args=(n,))
                # t.start()

    def disable_model(self, name: str | list):
        def disable_model_thread(name):
            if name not in self.__models:
                raise ValueError("Model not found")
            if self.__models[name].is_initialized():
                # print("Deinitializing", name)
                self.__models[name].deinitiate()

        if isinstance(name, str):
            disable_model_thread(name)
        elif isinstance(name, list):
            for n in name:
                disable_model_thread(n)
                # t = Thread(target=disable_model_thread, args=(n,))
                # t.start()

    def enable_model(self, name: str | list):
        def enable_model_thread(name):
            if name not in self.__models:
                raise ValueError("Model not found")
            if not self.__models[name].is_initialized():
                # print("Initializing", name)
                self.__models[name].initiate()

        if isinstance(name, str):
            enable_model_thread(name)
        elif isinstance(name, list):
            for n in name:
                enable_model_thread(n)
                # t = Thread(target=enable_model_thread, args=(n,))
                # t.start()

    def set_state_models(self, state_models: list):
        models = self.get_model_names()
        diff = [model for model in models if model not in state_models]

        self.disable_model(diff)
        if state_models != []:
            self.enable_model(state_models)

    def get_model_names(self):
        return list(self.__models.keys())

    def get_model_output_type(self, name):
        if name not in self.__models:
            raise ValueError("Model not found")
        return self.__models[name].get_output_type()

    def process_model(self, name, input_data):
        if name not in self.__models:
            raise ValueError("Model not found")
        return self.__models[name].process(input_data)

    def process(self, input_data: dict) -> dict:
        def process_thread(results, name, model, input_data):
            if model.is_initialized():
                results[name] = model.process(input_data)
            else:
                results[name] = None

        results = {}

        # threads = []
        for name, model in self.__models.items():
            process_thread(results, name, model, input_data)
        #     t = Thread(target=process_thread, args=(results, name, model, input_data))
        #     t.start()
        #     threads.append(t)

        # for t in threads:
        #     t.join()

        return results

    def draw_results(self, input_data, processed_results):
        def draw_results_thread(results, name, model, input_data, processed_results):
            processed_result = processed_results[name]
            if processed_result is None:
                results[name] = np.zeros((430, 640, 3), dtype=np.uint8)
            if model.is_initialized():
                results[name] = model.draw_results(input_data, processed_result)

        results = {}
        # threads = []
        for name, model in self.__models.items():
            draw_results_thread(results, name, model, input_data, processed_results)
        #     t = Thread(
        #         target=draw_results_thread,
        #         args=(results, name, model, input_data, processed_results),
        #     )
        #     threads.append(t)
        #     t.start()

        # for t in threads:
        #     t.join()

        return results
