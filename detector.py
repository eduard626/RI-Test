import torch
import os
from PIL import Image
import numpy as np
from torchvision.ops import box_convert
import cv2

from grounding_dino.groundingdino.util.inference import load_model, predict, load_image

# reuse transforms from GroundingDINO
import grounding_dino.groundingdino.datasets.transforms as T

# @TODO: We could load the model more gracefully, but ain't nobody got time for that.
GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"


class GdDetector:
    """
    A simple wrapper for the GroundingDINO model to perform object detection
    """

    def __init__(self):
        self.model = None

    def load_model(
        self,
        config_path: str = GROUNDING_DINO_CONFIG,
        checkpoint_path: str = GROUNDING_DINO_CHECKPOINT,
    ):
        """Loads the GroundingDINO model with the specified configuration and checkpoint.

        Args:
            config_path (str, optional): Path to config file. Defaults to GROUNDING_DINO_CONFIG.
            checkpoint_path (str, optional): Path to checkpoint point. Defaults to GROUNDING_DINO_CHECKPOINT.

        Raises:
            FileNotFoundError: If the specified config or checkpoint file does not exist.
            FileNotFoundError: If the specified checkpoint file does not exist.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        self.model = load_model(
            model_config_path=config_path,
            model_checkpoint_path=checkpoint_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def _predict(
        self,
        image: torch.Tensor,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> tuple:
        """
        Performs prediction using the GroundingDINO model on a given image and text prompt.

        Args:
            image (torch.Tensor): The input image tensor to be processed.
            text_prompt (str): The text prompt for grounding the objects in the image.
            box_threshold (float, optional): Threshold for box detection. Defaults to 0.25.
            text_threshold (float, optional): Threshold for text detection. Defaults to 0.25.

        Returns:
            tuple: A tuple containing:
                - boxes (np.ndarray): Detected bounding boxes.
                - logits (torch.Tensor): Logits for the detected boxes.
                - phrases (list): List of phrases corresponding to the detected boxes.
        """
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            remove_combined=True,  # we don't want combined stuff
        )
        # Convert boxes to pixel coordinates
        # image is C, H, W
        boxes = boxes * torch.Tensor(
            [
                image.shape[2],
                image.shape[1],
                image.shape[2],
                image.shape[1],
            ]
        )
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xywh").numpy()
        return boxes, logits, phrases

    def prep_image(self, image: Image.Image) -> torch.Tensor:
        """
        Prepares the image for GroundingDINO model inference.
        Uses the same transforms as in the GroundingDINO codebase.

        Args:
            image (Image.Image): The input image to be prepared

        Returns:
            torch.Tensor: The image tensor ready for model inference
        """
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_t, _ = transform(image, None)
        return image_t

    def predict(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ) -> tuple:
        """
        Predicts bounding boxes and phrases from an image file(path).

        Args:
            image_path (str): Path to the input image file.
            text_prompt (str): The text prompt for grounding.
            box_threshold (float, optional): Threshold for box detection. Defaults to 0.25.
            text_threshold (float, optional): Threshold for text detection. Defaults to 0.25.

        Raises:
            RuntimeError: If the model is not loaded before calling this method.
            FileNotFoundError: If the specified image file does not exist.

        Returns:
            tuple: Detected boxes, logits, and phrases.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        pil_image = Image.open(image_path).convert("RGB")
        image_tensor = self.prep_image(pil_image)
        return self._predict(image_tensor, text_prompt, box_threshold, text_threshold)

    def predict_np(
        self,
        image_np: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
    ):
        """
        Similar to above, but accepts a numpy array as input.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if image_np.ndim == 2:  # If grayscale
            raise ValueError("Input image must be a 3-channel RGB image.")
        # looks like blurring makes the predictions slightly more stable,
        image_np = cv2.GaussianBlur(image_np, (7, 7), 0)
        # assuming image is in RGB format !!!
        pil_image = Image.fromarray(image_np)
        image_tensor = self.prep_image(pil_image)
        return self._predict(image_tensor, text_prompt, box_threshold, text_threshold)
