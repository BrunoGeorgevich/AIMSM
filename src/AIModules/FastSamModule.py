from src.AIModules.AIModule import AIModule, ModuleOutput
from typing import Any
import numpy as np
import torch
import sys
import gc
import os

os.environ["YOLO_VERBOSE"] = "false"

sys.path.append("fastsam")
from fastsam import FastSAM, FastSAMPrompt


class FastSamModule(AIModule):
    """This class is the implementation of the model FastSAM as an AIModule"""

    __model = None
    __initialized = False
    __device = "cuda:0"

    def initiate(self, model_path: str = "weights/FastSAM-x.pt") -> None:
        """
        Initializes the object by loading the FastSAM model from the specified path. If no path is provided, the default path is "weights/FastSAM-x.pt".

        :param model_path: A string representing the path to the FastSAM model file.
        :type model_path: str
        :return: None"""
        self.__model = FastSAM(model_path)
        self.__initialized = True

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
    def process(self, input_data: dict) -> (FastSAMPrompt, Any):
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

            everything_results = self.__model(
                image,
                device=self.__device,
                retina_masks=True,
                imgsz=1024,
                conf=0.8,
                iou=0.5,
            )
            prompt_process = FastSAMPrompt(
                image, everything_results, device=self.__device
            )
            ann = prompt_process.everything_prompt()
            return prompt_process, ann

    @torch.no_grad()
    def draw_results(
        self, input_data: dict, results: (FastSAMPrompt, Any)
    ) -> np.ndarray:
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
        image = input_data.get("image", None)

        if image is None:
            raise ValueError("Image is not provided")

        if results is None:
            return None

        prompt = results[0]
        ann = results[1]

        try:
            output = prompt.plot_to_result(ann, retina=True)
        except IndexError:
            output = image

        del prompt
        del ann
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

        return ModuleOutput.IMAGE.name
