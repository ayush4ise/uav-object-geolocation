import tensorflow_hub as hub
import tensorflow as tf
import time
import numpy as np
from PIL import Image, ImageColor, ImageFont, ImageDraw
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import math
from exiftool import ExifTool
import os

# Importing functions from utils.py
from utils import (display_image, load_img, save_to_excel, image_details,
                   degrees_to_radians, rhumb_destination, assign_latitude_longitude, 
                   draw_bounding_box_on_image, draw_boxes, save_img)

def run_detector(detector, path):
    """
    Runs the object detector on the input image, draws bounding boxes around detected objects,
    and saves the annotated image and detection details.

    Parameters:
    detector (tf.Module): The TensorFlow Hub object detection module.
    path (str): The path to the input image.
    """
    img = load_img(path)
    image_height, image_width, _ = img.shape
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    result = detector(converted_img)
    end_time = time.time()
    result = {key: value.numpy() for key, value in result.items()}
    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time - start_time)

    image_with_boxes, detected_objects = draw_boxes(
        img.numpy(), result["detection_boxes"], result["detection_class_entities"], result["detection_scores"], path)

    display_image(image_with_boxes)
    save_img(image_with_boxes, "output/annotated_image.jpg")
    save_to_excel(detected_objects)

if __name__ == "__main__":
    module_handle = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
    detector = hub.load(module_handle).signatures['default']

    input_image_path = "data/input_image.jpg"  # Replace with your image path
    run_detector(detector, input_image_path)