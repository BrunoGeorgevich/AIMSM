from threading import Thread


class CSSM:
    def __init__(self):
        self.__states = {}
        self.__current_state = None

    def add_state(self, name: str):
        if name in self.__states:
            raise ValueError("State already exists")
        self.__states[name] = []

    def remove_state(self, name: str):
        if name not in self.__states:
            raise ValueError("State not found")
        del self.__states[name]

    def bind(self, name: str, model_name: str | list):
        def bind_model(name, model_name):
            if name not in self.__states:
                raise ValueError("State not found")

            if model_name in self.__states[name]:
                raise ValueError("Model already bound")

            self.__states[name].append(model_name)

        if isinstance(model_name, str):
            bind_model(name, model_name)
        elif isinstance(model_name, list):
            for n in model_name:
                t = Thread(target=bind_model, args=(name, n))
                t.start()

    def release(self, name: str, model_name: str):
        if name not in self.__states:
            raise ValueError("State not found")

        if model_name not in self.__states[name]:
            raise ValueError("Model not found")

        self.__states[name].remove(model_name)

    def switch(self, name: str):
        if name not in self.__states:
            raise ValueError("State not found")

        if self.__current_state is None:
            self.__current_state = name
        else:
            if self.__current_state == name:
                return
            else:
                self.__current_state = name

    def get_states(self):
        return list(self.__states.keys())

    def state_models(self):
        if self.__current_state is None:
            raise ValueError("No state defined")
        if self.__current_state not in self.__states:
            raise ValueError("State not found")
        return self.__states[self.__current_state]

    def current_state(self):
        return self.__current_state
