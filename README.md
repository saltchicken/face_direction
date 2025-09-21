This project is a simple but powerful tool for analyzing the orientation of a human face in an image. Using MediaPipe's Face Mesh, it detects key facial landmarks and employs computer vision techniques to determine the face's yaw, pitch, and roll angles. It then translates these technical angles into intuitive directional commands like "up" or "left."

🚀 Features

    Face Orientation Analysis: Accurately calculates yaw, pitch, and roll angles of a face.

    Directional Interpretation: Translates angles into simple, human-readable directions (e.g., "right," "down," or "straight").

    Visualization: (Optional) Visualize the face landmarks and a 3D coordinate system to see the orientation in real-time.

⚙️ Installation

First, clone the repository:
Bash

git clone https://github.com/your-username/face-direction.git
cd face-direction

Next, install the required packages. This project uses setuptools, so the dependencies are managed by pyproject.toml.
Bash

pip install .

📖 Usage

Command Line Interface

You can run the project directly from your terminal.
Bash

face_direction <path_to_image> [--show]

    path_to_image: The path to the image file you want to analyze.

    --show: An optional flag that displays the image with the projected 3D axes and landmarks. This is a great way to visually confirm the face's orientation.

Example

To analyze an image named test_face.jpg and see the visualization:
Bash

face_direction test_face.jpg --show

The output will provide the calculated yaw and pitch angles, followed by the interpreted direction.

Yaw: 15.34 degrees, Pitch: -5.67 degrees
Direction: right straight

👩‍💻 Project Structure

The project is structured to be clean and modular.

.
├── pyproject.toml
└── src/
    └── face_direction/
        ├── __init__.py
        ├── __main__.py          # Entry point for the CLI
        ├── dataclasses.py       # Defines the Direction dataclass
        ├── face_orienter.py     # Main class for face detection and orientation
        └── utils.py             # Helper functions for calculations and drawing

    face_orienter.py: Contains the core logic for processing images, finding landmarks, and calculating orientation.

    dataclasses.py: A simple dataclass to store and interpret the face's yaw and pitch values into an easy-to-understand format.

    utils.py: A collection of reusable functions, keeping the main FaceOrienter class focused on its primary task.
