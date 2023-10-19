from typing import Any, Protocol
import numpy as np


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

    def process(self, image: np.ndarray) -> Any:
        """
        The function "process" takes an image as input and process it.

        :param image: The image parameter is a numpy array representing an image
        :type image: np.ndarray
        :return: The function "process" returns the results of the processing
        :rtype: Any
        """
        ...

    def draw_results(self, image: np.ndarray, results: Any) -> np.ndarray:
        """
        The function "draw_results" takes an image and results as input and draws the results.

        :param image: The image parameter is a numpy array representing an image. It can be a grayscale
        image or a color image with multiple channels
        :type image: np.ndarray
        :param results: The "results" parameter is a variable that contains the results of some
        computation or analysis. It can be of any data type, depending on the specific use case
        :type results: Any
        :return: The function "draw_results" returns the image with the results drawn on it
        :rtype: np.ndarray
        """
        ...
