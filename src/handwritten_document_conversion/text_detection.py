import os
from ultralytics import YOLO
import cv2


file_path = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(file_path, '..', '..' ,'models')
model_path = os.path.join(MODEL_DIR, 'best.pt')

OG_IMG_DIR = os.path.join(file_path, '..', '..', 'images', 'original')
RESIZED_IMG_DIR = os.path.join(file_path, '..', '..', 'images', 'resized')


class TextDetection:
    _model = None

    def __init__(self, image_path) -> None:
        self.image_path = image_path

        if TextDetection._model is None:
            TextDetection._model = YOLO(model_path)
        
    def detect(self):
        """
        Function to return the results
        """
        results = TextDetection._model(self.image_path)
        return results

    def return_bboxes(self):
        """
        Returns the bounding boxes of the detected texts
        """
        results = self.detect()
        bboxes = []
        for result in results:
            boxes = result.boxes.data.tolist()
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        return bboxes

    def return_cropped_images(self):
        """
        Returns the cropped images based on the bounding boxes, sorted left to right
        """
        # Read the image
        image = cv2.imread(self.image_path)

        # Get bounding boxes
        bboxes = self.return_bboxes()

        # Sort bounding boxes from left to right based on the x1 coordinate
        bboxes = sorted(bboxes, key=lambda x: x[0])

        # Crop images
        cropped_images = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)

        # Display the cropped images
        for idx, cropped_img in enumerate(cropped_images):
            file_name = f"cropped_img {idx+1}.jpg"
            cv2.imwrite(os.path.join(RESIZED_IMG_DIR, file_name), cropped_img)
            cv2.imshow(file_name, cropped_img)
            cv2.waitKey(0)  # Wait for a key press to close the image window
            cv2.destroyAllWindows()

        return cropped_images
