import tensorflow as tf
import numpy as np
from PIL import Image, ImageColor, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import math
from exiftool import ExifTool

# # Function to calculate Euclidean distance between two points
# def calculate_distance(x1, y1, x2, y2):
#     # Calculate the Euclidean distance between two points
#     return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Function to display the image
def display_image(image):
    """
    Displays the given image using matplotlib.
    
    Parameters:
    image (numpy.ndarray): The image to be displayed.
    """
    fig = plt.figure(figsize=(20, 15))
    plt.grid(False)
    plt.imshow(image)

# Function to load the image
def load_img(path):
    """
    Loads an image from the given file path.

    Parameters:
    path (str): The path to the image file.

    Returns:
    numpy.ndarray: The loaded image.
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

# Function to save the annotated image
def save_img(image, path):
    """
    Saves the given image to the specified path.

    Parameters:
    image (numpy.ndarray): The image to be saved.
    path (str): The path where the image will be saved.
    """
    image = Image.fromarray(image)
    image.save(path)
    print(f"Image saved at {path}")

# Function to save the coordinates of detected objects to an excel file
def save_to_excel(detected_objects):
    """
    Saves the detected objects' coordinates to an Excel file.

    Parameters:
    detected_objects (list): A list of dictionaries containing the details of detected objects.
    """
    df = pd.DataFrame(detected_objects)
    df.to_excel("output/detected_objects.xlsx", index=False)
    print("Detected objects saved to output/detected_objects.xlsx")

# Function to extract information from the image
def image_details(image_path):
    """
    Extracts details from the image using ExifTool.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    dict: A dictionary containing image details like relative altitude, field of view, camera yaw, centre latitude, and centre longitude.
    """
    with ExifTool("C:\Program Files\ExifTool\exiftool.exe") as et:
        # Relative altitude of the drone to the ground
        rel_altitude = float(et.execute("-XMP-drone-dji:RelativeAltitude", image_path)[50:56])
        # Field of view of the camera
        fov = float(et.execute("-Composite:FOV", image_path)[50:56])
        # Camera yaw (rotation around the vertical axis)
        camera_yaw = et.execute_json(*["-MakerNotes:All"] + [image_path])[0]['MakerNotes:CameraYaw']
        # Centre latitude of the image
        centre_latitude = float(et.execute("-Composite:GPSLatitude", image_path)[50:-2])
        # Centre longitude of the image
        centre_longitude = float(et.execute("-Composite:GPSLongitude", image_path)[50:-2])

    return {
        'Relative Altitude': rel_altitude,
        'Field of View': fov,
        'Camera Yaw': camera_yaw,
        'Centre Latitude': centre_latitude,
        'Centre Longitude': centre_longitude
    }

# Function to convert degrees to radians
def degrees_to_radians(degrees):
    """
    Converts degrees to radians.

    Parameters:
    degrees (float): Angle in degrees.

    Returns:
    float: Angle in radians.
    """
    return degrees * (math.pi / 180)

# Functipn for rhumb destination
def rhumb_destination(start_lat, start_long, bearing, distance):
    """
    Calculates the destination point from a given point, bearing, and distance using the rhumb line formula.

    Parameters:
    start_lat (float): Starting latitude.
    start_long (float): Starting longitude.
    bearing (float): Bearing in degrees.
    distance (float): Distance in meters.

    Returns:
    list: A list containing the latitude and longitude of the destination point.
    """
    # Returns the destination point from a given point and bearing
    # using the rhumb line formula
    R = 6378137
    delta = distance / R  # d = angular distance covered on earth's surface

    lambda_1 = start_long * math.pi / 180

    phi_1 = degrees_to_radians(start_lat)
    theta = degrees_to_radians(bearing)

    delta_phi = delta * math.cos(theta)
    phi_2 = phi_1 + delta_phi

    # check for some points going past the pole, normalise latitude if so
    if abs(phi_2) > (math.pi / 2) and (phi_2 > 0):
        phi_2 = math.pi - phi_2
    if abs(phi_2) > (math.pi / 2) and (phi_2 < 0):
        phi_2 = math.pi - phi_2

    delta_psi = math.log(math.tan(phi_2 / 2 + math.pi / 4) / math.tan(phi_1 / 2 + math.pi / 4))

    # E-W course becomes ill-conditioned with 0/0
    if abs(delta_psi) > 10e-12:
        q_1 = delta_phi / delta_psi
    else:
        q_1 = math.cos(phi_1)

    delta_lambda = delta * math.sin(theta) / q_1
    lambda_2 = lambda_1 + delta_lambda

    # normalise to −180..+180° 
    destination = [
        (phi_2 * 180 / math.pi),
        math.fmod(((lambda_2 * 180 / math.pi) + 540), 360) - 180,
    ]
    # (latitude, longitude) format
    return destination

# Function to assign latitude and longitude based on the distance from the center of the image
def assign_latitude_longitude(coordinates, image_height, image_width, image_path):
    """
    Assigns latitude and longitude to detected objects based on their distance from the center of the image.

    Parameters:
    coordinates (tuple): Tuple containing the x and y coordinates of the detected object.
    image_height (int): Height of the image.
    image_width (int): Width of the image.
    image_path (str): Path to the image file.

    Returns:
    tuple: A tuple containing the latitude, longitude, and distance of the detected object.
    """
    # getting info from image
    details = image_details(image_path)

    original_latitude = details['Centre Latitude']
    original_longitude = details['Centre Longitude']
    altitude = details['Relative Altitude'] * 0.3048 # convert to meters

    fov = details['Field of View']

    # Field of view of the camera in radians
    fov = fov * math.pi / 180
    # Field of view in the vertical direction
    fovAtan = math.tan(fov)

    # calculate the ground distance shown (diagonal distance from top-left to bottom-right corner)
    diagonalDistance = altitude * fovAtan

    # the direction the drone is pointed
    camerayaw = details['Camera Yaw']
    # Normalize to 0 to 360 degrees
    camerayaw = (camerayaw + 360) % 360

    # the bearing of the object relative to the drone
    bearing = (camerayaw - 90) % 360

    # change coordinate system so the center point of the image is (0, 0) (instead of the top-left point)
    # this means that (0, 0) is where our drone is and makes our math easier

    ############################
    normalized = (coordinates[1] - image_height / 2, coordinates[0] - image_width / 2)
    ############################
    # normalized = (coordinates[0] - image_width / 2, coordinates[1] - image_height / 2)

    # calculate the distance and bearing of the object relative to the center point
    distanceFromCenterInPixels = math.sqrt(normalized[0]**2 + normalized[1]**2)
    diagonalDistanceInPixels = math.sqrt(image_width*image_width + image_height*image_height)

    percentOfDiagonal = distanceFromCenterInPixels / diagonalDistanceInPixels
    distance = percentOfDiagonal * diagonalDistance # in meters

    # calculate the angle of the object relative to the center point
    angle = math.atan(normalized[0]/(normalized[1]+0.000001)) * 180 / math.pi

    # if the detection is in the right half of the frame we need to rotate it 180 degrees
    if normalized[1] >= 0:
        angle += 180
   
    # use that distance and bearing to get the GPS location of the panel
    point = rhumb_destination(original_latitude, original_longitude, (bearing + angle)%360, distance)

    return point[0], point[1], distance

# Function to draw a bounding box on the image
def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=4, display_str_list=()):
    """
    Draws a bounding box on the image.

    Parameters:
    image (PIL.Image.Image): The image to draw on.
    ymin (float): The ymin coordinate of the bounding box.
    xmin (float): The xmin coordinate of the bounding box.
    ymax (float): The ymax coordinate of the bounding box.
    xmax (float): The xmax coordinate of the bounding box.
    color (str): The color of the bounding box.
    font (PIL.ImageFont.ImageFont): The font to use for the display strings.
    thickness (int, optional): The thickness of the bounding box. Defaults to 4.
    display_str_list (list, optional): A list of strings to display in the bounding box. Defaults to ().
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

    display_str_heights = [font.getbbox(ds)[3] for ds in display_str_list]
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    for display_str in display_str_list[::-1]:
        bbox = font.getbbox(display_str)
        text_width, text_height = bbox[2], bbox[3]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
        text_bottom -= text_height - 2 * margin

# Function to draq boxes around detected objects
def draw_boxes(image, boxes, class_names, scores, image_path):
    """
    Draws bounding boxes around detected objects and annotates them with class names, scores, latitude, longitude, and distance.

    Parameters:
    image (numpy.ndarray): The image to draw on.
    boxes (numpy.ndarray): The bounding boxes of detected objects.
    class_names (list): The class names of detected objects.
    scores (list): The confidence scores of detected objects.
    image_path (str): The path to the image file.

    Returns:
    tuple: A tuple containing the annotated image and a list of detected objects with their details.
    """
    # pre-set values
    max_boxes=10
    min_score=0.1

    colors = list(ImageColor.colormap.values())
    try:
        font_size = 12  # Adjust the font size as needed
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    detected_objects = []
    image_height, image_width, _ = image.shape  # Get image dimensions
    # image_center_x = image_width // 2
    # image_center_y = image_height // 2

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            object_center_x = (xmin + xmax) / 2 * image_width
            object_center_y = (ymin + ymax) / 2 * image_height

            latitude, longitude, distance = assign_latitude_longitude((object_center_x, object_center_y), image_height, image_width, image_path)

            display_str = "{}: {}% Lat: {:.2f}, Long: {:.2f}, Dist: {:.2f}".format(
                class_names[i].decode("ascii"), int(100 * scores[i]), latitude, longitude, distance)
            color = colors[hash(class_names[i]) % len(colors)]
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            draw_bounding_box_on_image(
                image_pil,
                ymin,
                xmin,
                ymax,
                xmax,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
            detected_objects.append({
                "class_name": class_names[i].decode("ascii"),
                "score": scores[i],
                "latitude": latitude,
                "longitude": longitude,
                "distance": distance
            })
    return image, detected_objects