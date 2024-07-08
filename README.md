# 📸 UAV Object Geolocation README

Welcome to the UAV Object Geolocation project! This repository contains code for detecting objects in images captured by UAVs (drones) using an object detection model and extracting their geolocation data based on image metadata.


##  Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [Documentation](#documentation)
- [License](#license)

## Introduction

This project uses a TensorFlow object detection model to identify objects in UAV images. Additionally, it extracts geolocation information from the image's EXIF data to assign geographic coordinates to the detected objects. 

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ayush4ise/uav-object-geolocation.git
    cd uav-object-geolocation
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have ExifTool installed. You can download it from [here](https://exiftool.org/).

## Usage

1. Place your input image in the `data` directory as `input_image.jpg`.
2. Run the detection script:
    ```bash
    python src/main.py
    ```

## Demo

To see the project in action, you can run the provided Jupyter notebook:
```bash
jupyter notebook object_detection_demo.ipynb
```

## Documentation

For detailed documentation on the project specifics, including model details, calculation methods, and more, refer to the [documentation.md](docs/documentation.md) file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributions

We welcome contributions! Please fork the repository and submit pull requests with clear descriptions of your changes.

## Contact

For any questions or feedback, please reach out via [e-mail](mailto:ayush.parm.ise@gmail.com).