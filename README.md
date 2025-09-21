# Face Direction

A Python package to detect and analyze a person's face direction (yaw, pitch, and roll) from a still image using OpenCV and MediaPipe. 🧐

---

## 🚀 Features

* **Face Orientation Detection:** Accurately calculates the yaw, pitch, and roll angles of a face.
* **3D Visualization:** Overlays a 3D coordinate system on the detected face to visually represent its orientation.
* **Simple Command-Line Interface:** Easily analyze an image from the terminal.
* **Modular Code:** Separates core logic, utility functions, and data structures for clarity.

---

## 🛠️ Installation

You can install `face_direction` directly from this repository.

**Install the package:**
    
```sh
    pip install git+https://github.com/saltchicken/face_direction
```
    
This command will install the package and its dependencies, including `opencv-python` and `mediapipe`.

---

## 💻 Usage

### Command Line Interface

You can use the installed `face_direction` command to analyze an image.

* **Basic Analysis:**
    ```sh
    face_direction <path-to-your-image.jpg>
    ```
    This will print the calculated yaw and pitch angles and the detected direction (e.g., "right up" or "straight").

* **With Visualization:**
    Add the `--show` flag to display the image with the 3D axis visualization.
    ```sh
    face_direction <path-to-your-image.jpg> --show
    ```

### Python Module

You can also import and use the `FaceOrienter` class in your own Python scripts.

```python
from face_direction.face_orienter import FaceOrienter

# Initialize the face orienter
face_orienter = FaceOrienter()

# Analyze an image
yaw, pitch, roll = face_orienter.orient("path/to/your/image.jpg", show=True)

if yaw is not None:
    print(f"Yaw: {yaw:.2f} degrees")
    print(f"Pitch: {pitch:.2f} degrees")
    print(f"Roll: {roll:.2f} degrees")
