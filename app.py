import os
import gradio as gr
from src.handwritten_document_conversion.text_recognition import TextRecognition
from src.handwritten_document_conversion.text_detection import TextDetection

from utils.image_processing import PreprocessImage

file_path = os.path.dirname(os.path.abspath(__file__))
RESIZED_IMG_DIR = os.path.join(file_path, 'images', 'resized')

def pipeline_function(img):
    """
    Function to return text of the whole image file associated
    :param img: filepath
    :return: final_text
    """
    text_det_obj = TextDetection(image_file=img)
    text_rec_obj = TextRecognition()

    _, cropped_images_file_name = text_det_obj.return_cropped_images()

    texts = []
    for cropped_image in cropped_images_file_name:
        generated_text = text_rec_obj.return_generated_text(image_path=os.path.join(RESIZED_IMG_DIR, cropped_image))
        texts.append(generated_text)

    # Combine all the recognized text
    final_text = ' '.join(texts)
    return final_text

# Gradio interface
iface = gr.Interface(
    fn=pipeline_function,
    inputs=gr.Image(type="filepath", label="Upload your handwritten images"),  # File upload input
    outputs="text",  # Display recognized text
    title="Handwritten Document Conversion - Nepali",
    description="Upload a page of Nepali handwritten text to extract and scan text.",
    examples=["images/original/test_text.png", "images/original/kabita.jpg"],  # Example images for demonstration
)

if __name__ == "__main__":
    iface.launch()
