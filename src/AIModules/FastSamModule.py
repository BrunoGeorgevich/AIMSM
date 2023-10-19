from src.AIModules.AIModule import AIModule
from typing import Any
import numpy as np
import sys

sys.path.append("fastsam")
from fastsam import FastSAM, FastSAMPrompt


class FastSamModule(AIModule):
    """This class is the implementation of the model FastSAM as an AIModule"""

    __model = None  # The variable that will hold the FastSAM model
    __device = "cuda:0"  # The model will be executed on this device

    def initiate(self, model_path: str = "weights/FastSAM-x.pt") -> None:
        """
        Initializes the object by loading the FastSAM model from the specified path. If no path is provided, the default path is "weights/FastSAM-x.pt".

        :param model_path: A string representing the path to the FastSAM model file.
        :type model_path: str
        :return: None"""
        self.__model = FastSAM(model_path)

    def deinitiate(self) -> None:
        """Deinitializes the FastSAM model"""
        self.__model = None

    def process(self, image: np.ndarray) -> (FastSAMPrompt, Any):
        """
        The function processes an image using a model and returns a prompt process object and the
        annotations generated from the prompt process.

        :param image: The `image` parameter is an input image represented as a NumPy array
        (`np.ndarray`)
        :type image: np.ndarray
        :return: a tuple containing two values: `prompt_process` and `ann`.
        """
        if self.__model is None:
            raise ValueError("Model is not initiated")

        everything_results = self.__model(
            image,
            device=self.__device,
            retina_masks=True,
            imgsz=1024,
            conf=0.8,
            iou=0.5,
        )
        prompt_process = FastSAMPrompt(image, everything_results, device=self.__device)
        ann = prompt_process.everything_prompt()
        return prompt_process, ann

    def draw_results(
        self, image: np.ndarray, results: (FastSAMPrompt, Any)
    ) -> np.ndarray:
        """
        The function takes an image and a list of results, and returns the image with the prompt and
        annotation plotted on it, or the original image if there are no results.

        :param image: The `image` parameter is the input image on which the results will be drawn
        :type image: np.ndarray
        :param results: The `results` parameter is a tuple that contains two elements, the FastSAMPrompt and the annotations
        :type results: (FastSAMPrompt, Any)
        :return: an image with the segmented area drawn on it
        """
        prompt = results[0]
        ann = results[1]

        try:
            return prompt.plot_to_result(ann, retina=True, mask_random_color=False)
        except IndexError:
            return image
