import os
import cv2
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor, TrOCRProcessor, VisionEncoderDecoderModel
from dotenv import load_dotenv
import numpy as np
import torch  # Import PyTorch for device handling

load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME')
FEATURE_EXTRACTOR = os.getenv('FEATURE_EXTRACTOR')


class TextRecognition:
    _tokenizer = None
    _model = None
    _feature_extractor = None
    _processor = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability

    def __init__(self):
        if TextRecognition._tokenizer is None:
            TextRecognition._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if TextRecognition._model is None:
            TextRecognition._model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
            TextRecognition._model.config.early_stopping = True
            TextRecognition._model.to(TextRecognition.device)  # Move the model to the specified device

        if TextRecognition._feature_extractor is None:
            TextRecognition._feature_extractor = ViTImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        if TextRecognition._processor is None:
            TextRecognition._processor = TrOCRProcessor(feature_extractor=TextRecognition._feature_extractor,
                                                        tokenizer=TextRecognition._tokenizer
                                                        )

    @staticmethod
    def return_generated_text(image_path: str) -> str:
        """
        Function to return text associated with each cropped image file
        :param image_path: OpenCV image (NumPy array)
        :return: generated_text
        """

        # Convert OpenCV image (NumPy array) to PIL Image
        pil_image = Image.open(image_path)

        if TextRecognition._processor is None:
            raise ValueError("Processor is not initialized.")

        # Process the image
        pixel_values = TextRecognition._processor(pil_image, return_tensors="pt").pixel_values

        # Move pixel values to the specified device
        pixel_values = pixel_values.to(TextRecognition.device)

        # Generate text
        generated_ids = TextRecognition._model.generate(pixel_values, early_stopping=True)
        generated_text = TextRecognition._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text