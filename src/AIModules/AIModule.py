from typing import Any, Protocol
from enum import Enum
import numpy as np


class ModuleOutput(Enum):
    IMAGE = 1
    TEXT = 2


class AIModule(Protocol):
    """This class represents an AIModule and defines the basic methods that all modules must implement"""

    def initiate(self, model_path: str) -> None:
        """
        The `initiate` function initializes the model by using the specified model path.

        :param model_path: The `model_path` parameter is a string that represents the file path to the
        model that you want to load or initialize
        :type model_path: str
        """
        ...

    def deinitiate(self) -> None:
        """
        The `deinitiate` function deinitializes the model.
        """
        ...

    def process(self, input_data: dict) -> Any:
        """
        The function "process" takes an image as input and process it.

        :param input_data: Input data dictionary, which contains all the data collected as input.
        :type input_data: dict
        :return: The function "process" returns the results of the processing
        :rtype: Any
        """
        ...

    def draw_results(self, input_data: dict, results: Any) -> np.ndarray:
        """
        The function "draw_results" takes an image and results as input and draws the results.

        :param input_data: Input data dictionary, which contains all the data collected as input.
        :type input_data: dict
        :param results: The "results" parameter is a variable that contains the results of some
        computation or analysis. It can be of any data type, depending on the specific use case
        :type results: Any
        :return: The function "draw_results" returns the image with the results drawn on it
        :rtype: np.ndarray
        """
        ...

    def is_initialized(self) -> bool:
        """
        The function returns the value of the __initialized attribute.

        :return: a boolean value indicating whether the model is initialized or not.
        :rtype: bool
        """
        ...

    def get_output_type(self) -> str:
        """
        The function returns the type of the output of the module.

        :return: a string indicating the type of the output of the module.
        :rtype: str
        """
        ...
