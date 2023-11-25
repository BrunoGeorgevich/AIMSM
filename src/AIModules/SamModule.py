from src.AIModules.AIModule import AIModule, ModuleOutput
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import cv2
import gc

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class SamModule(AIModule):
    """This class is the implementation of the model FastSAM as an AIModule"""

    __model = None
    __mask_generator = None
    __initialized = False
    __device = "cuda:0"
    __colors = []

    @torch.no_grad()
    def initiate(self, model_path: str = "weights/vit_b.pth") -> None:
        """
        Initializes the object by loading the FastSAM model from the specified path. If no path is provided, the default path is "weights/FastSAM-x.pt".

        :param model_path: A string representing the path to the FastSAM model file.
        :type model_path: str
        :return: None
        """
        self.__model = sam_model_registry["vit_b"](checkpoint=model_path)
        self.__model.to(device=self.__device)
        self.__mask_generator = SamAutomaticMaskGenerator(self.__model)
        self.__initialized = True

    @torch.no_grad()
    def deinitiate(self) -> None:
        """Deinitializes the FastSAM model"""
        del self.__model
        del self.__mask_generator
        self.__model = None
        self.__mask_generator = None
        self.__initialized = False

        try:
            torch._C._cuda_clearCublasWorkspaces()
            torch._dynamo.reset()
        except AttributeError:
            pass

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def process(self, input_data: dict) -> Any:
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

            resized = cv2.resize(image, (512, 512))
            masks = self.__mask_generator.generate(resized)

            del resized
            gc.collect()
            torch.cuda.empty_cache()

            return masks

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

        def show_anns(image, anns):
            if len(anns) == 0:
                return image

            sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

            mask = np.zeros_like(image)
            for i, ann in enumerate(sorted_anns):
                m = ann["segmentation"]

                if i > len(self.__colors) - 1:
                    self.__colors.append((np.random.random(3) * 255).astype(np.uint8))

                mask[m] = self.__colors[i]

            combined = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

            del mask
            del image
            gc.collect()
            torch.cuda.empty_cache()

            return combined

        image = input_data.get("image", None)

        if image is None:
            raise ValueError("Image is not provided")

        del image
        gc.collect()
        torch.cuda.empty_cache()

        resized = cv2.resize(image, (512, 512))

        try:
            return show_anns(resized, results)
        except IndexError:
            return resized

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
