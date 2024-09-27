import os
from typing import List

import numpy as np
from ultralytics import YOLO
import cv2

from utils.image_processing import PreprocessImage


# Get the parent directory of the current Python file (handwritten_document_conversion)
file_path = os.path.dirname(os.path.abspath(__file__))

# Move up two directories from src/handwritten_document_conversion to the project root
root_dir = os.path.abspath(os.path.join(file_path, '..', '..'))

# Set the correct paths for models and images
MODEL_DIR = os.path.join(root_dir, 'models')
model_path = os.path.join(MODEL_DIR, 'best.pt')

OG_IMG_DIR = os.path.join(root_dir, 'images', 'original')
RESIZED_IMG_DIR = os.path.join(root_dir, 'images', 'resized')

class TextDetection:
    _model = None

    def __init__(self, image_file) -> None:
        self.image_file = image_file
        self.image_preprocess = PreprocessImage()

        if TextDetection._model is None:
            TextDetection._model = YOLO(model_path)

    def detect(self) -> list:
        """
        Function to return results from TextDetection Model
        :return: results
        """
        results = TextDetection._model(os.path.join(OG_IMG_DIR, self.image_file))
        return results

    def return_bboxes(self) -> list[list[int]]:
        """
        Function to return bounding box of each cropped image
        :return: bboxes
        """
        results = self.detect()
        bboxes = []
        for result in results:
            boxes = result.boxes.data.tolist()
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        return bboxes

    def return_cropped_images(self) -> (list[np.ndarray], list[str]):
        """
        Function to return cropped_images list and saved images file_path
        :return: cropped_images, cropped_images_file_name
        """
        # Read the image
        image = cv2.imread(os.path.join(OG_IMG_DIR, self.image_file))

        # Get bounding boxes
        bboxes = self.return_bboxes()

        bboxes_with_centers = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            bboxes_with_centers.append((bbox, (center_x, center_y)))

        # Sort bounding boxes first by the y-coordinate of the center, then by the x-coordinate (left to right)
        bboxes_with_centers = sorted(bboxes_with_centers, key=lambda x: (x[1][1], x[1][0]))

        # Crop images
        cropped_images = []
        cropped_images_file_name = []
        for idx, (bbox, center) in enumerate(bboxes_with_centers):
            x1, y1, x2, y2 = bbox
            cropped_image = image[y1:y2, x1:x2]

            cropped_image = self.image_preprocess.preprocess_image(cropped_image)   # Added padding to make size (224,224) before saving
            cropped_images.append(cropped_image)

            # Create a file name for the cropped image
            file_name = f"{os.path.splitext(self.image_file)[0]}_{idx + 1}{os.path.splitext(self.image_file)[-1]}"
            cropped_images_file_name.append(file_name)

            cv2.imwrite(os.path.join(RESIZED_IMG_DIR, file_name), cropped_image)

            # cv2.imshow(file_name, cropped_image)
            # cv2.waitKey(0)  # Wait for a key press to close the image window
            # cv2.destroyAllWindows()

        return cropped_images, cropped_images_file_name


