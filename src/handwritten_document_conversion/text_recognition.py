from PIL import Image
from transformers import TrOCRProcessor,VisionEncoderDecoderModel

MODEL_NAME = "paudelanil/denvagari-TrOCR"
MODEL_NAME_1 = "aayushpuri01/TrOCR-Devanagari"

class TextRecognition:
    _processor = None
    _model = None

    def __init__(self):
        if TextRecognition._processor is None:
            TextRecognition._processor = TrOCRProcessor.from_pretrained(MODEL_NAME_1)

        if TextRecognition._model is None:
            TextRecognition._model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME_1)

    @staticmethod
    def return_generated_text(image: Image) -> str:
        """
        Function to return text associated with each cropped image file
        :param image: PIL Image
        :return: generated_text
        """
        pil_image = Image.open(image)

        if TextRecognition._processor is None:
            raise ValueError("Processor is not initialized.")

        pixel_values = TextRecognition._processor(pil_image, return_tensors="pt").pixel_values
        generated_ids = TextRecognition._model.generate(pixel_values)
        generated_text = TextRecognition._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text



