from transformers import pipeline
from PIL import Image
from typing import Any
import numpy as np
import torch
import gc

from src.AIModules.AIModule import AIModule, ModuleOutput


class RoomClassificationModule(AIModule):
    """This class is the implementation of a model Room Classification AI Module"""

    __model = None  # The variable that will hold the Room Classification model
    __initialized = (
        False  # The variable that will indicate whether the model is initialized
    )
    __device = "cuda:0"  # The model will be executed on this device

    @torch.no_grad()
    def initiate(self, model_path: str = "HHousen/household-rooms") -> None:
        """
        Initializes the object by loading the FastSAM model from the specified path. If no path is provided, the default path is "weights/FastSAM-x.pt".

        :param model_path: A string representing the model card name to the Room Classification model file.
        :type model_path: str
        :return: None"""
        self.__model = pipeline(
            "image-classification", model_path, device=self.__device, torch_dtype=torch.float16
        )
        self.__initialized = True

    @torch.no_grad()
    def deinitiate(self) -> None:
        """Deinitializes the FastSAM model"""
        del self.__model
        self.__model = None
        self.__initialized = False

        try:
            torch._C._cuda_clearCublasWorkspaces()
            torch._dynamo.reset()
        except AttributeError:
            pass

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def process(self, input_data: dict) -> list[str]:
        """
        The function processes an image using a model and returns a prompt process object and the
        annotations generated from the prompt process.

        :param input_data: Input data dictionary, which must contains the image to be processed.
        {
            "image": np.ndarray
            ...
        }
        :return: a tuple containing two values: `prompt_process` and `ann`.
        """
        with torch.no_grad():
            image = input_data.get("image", None)

            if image is None:
                raise ValueError("Image is not provided")

            if self.__initialized is False:
                raise ValueError("Model is not initiated")

            image = Image.fromarray(image)
            preds = self.__model(image)
            best = preds[0]

            del preds
            del image
            gc.collect()
            torch.cuda.empty_cache()

            return best

    @torch.no_grad()
    def draw_results(self, input_data: dict, results: Any) -> np.ndarray:
        """
        The function takes an image and a list of results, and returns the image with the prompt and
        annotation plotted on it, or the original image if there are no results.

        :param input_data: Input data dictionary, which must contains the image to be processed.
        {
            "image": np.ndarray
            ...
        }
        :param results: The `results` parameter is a tuple that contains two elements, the FastSAMPrompt and the annotations
        :type results: (FastSAMPrompt, Any)
        :return: an image with the segmented area drawn on it
        """
        output = f"Room: {results['label']}\nConfidence: {results['score']*100:.2f}%"

        del results
        del input_data
        gc.collect()
        torch.cuda.empty_cache()

        return output

    def is_initialized(self) -> bool:
        """
        The function returns the value of the __initialized attribute.

        :return: a boolean value indicating whether the model is initialized or not.
        :rtype: bool
        """
        return self.__initialized

    def get_output_type(self) -> str:
        """
        The function returns the type of the output of the module.

        :return: a string indicating the type of the output of the module.
        :rtype: str
        """

        return ModuleOutput.TEXT.name
