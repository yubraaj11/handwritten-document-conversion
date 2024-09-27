import os
import cv2
from PIL import Image
from transformers import AutoTokenizer, ViTImageProcessor, TrOCRProcessor, VisionEncoderDecoderModel
from dotenv import load_dotenv
import numpy as np

load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME')
FEATURE_EXTRACTOR = os.getenv('FEATURE_EXTRACTOR')


class TextRecognition:
    _tokenizer = None
    _model = None
    _feature_extractor = None
    _processor = None

    def __init__(self):
        if TextRecognition._tokenizer is None:
            TextRecognition._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if TextRecognition._model is None:
            TextRecognition._model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
            TextRecognition._model.config.early_stopping = True

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
        pil_image = cv2.imread(image_path)

        if TextRecognition._processor is None:
            raise ValueError("Processor is not initialized.")

        # Process the image and generate text
        pixel_values = TextRecognition._processor(pil_image, return_tensors="pt").pixel_values
        generated_ids = TextRecognition._model.generate(pixel_values,
                                                        early_stopping=True)  # Pass early_stopping=True
        generated_text = TextRecognition._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text

