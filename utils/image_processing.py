import os 
import cv2



def resize_image_for_yolov8(image_path, target_size=640):
    """
    Function to return image of size (640, 640) for YOLOv8 compatibility
    :param image_path:
    :param target_size:
    :return:
    """
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (target_size, target_size))


    return resized_image
