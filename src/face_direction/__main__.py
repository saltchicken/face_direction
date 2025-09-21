import argparse
import cv2
import numpy as np

from .face_orienter import FaceOrienter
from .dataclasses import Direction

def main():
    """Main function to run the face direction detection."""
    parser = argparse.ArgumentParser(description="Analyze face direction from an image file.")
    parser.add_argument("input_file", help="Path to the input image file")
    parser.add_argument("--show", action="store_true", help="Show the visualization on the image")
    args = parser.parse_args()

    # Read the image from the file path and convert it to a NumPy array
    frame = cv2.imread(args.input_file)
    if frame is None:
        print(f"Error: Unable to load image from {args.input_file}")
        return

    face_orienter = FaceOrienter()
    yaw, pitch, roll = face_orienter.orient(frame, args.show)

    if yaw is not None and pitch is not None:
        face_direction = Direction(yaw, pitch)
        print(f"Yaw: {face_direction.yaw:.2f} degrees, Pitch: {face_direction.pitch:.2f} degrees")
        print(f"Direction: {face_direction}")
    else:
        print("Could not determine face direction.")

if __name__ == "__main__":
    main()
