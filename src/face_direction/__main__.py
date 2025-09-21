import argparse
from .face_orienter import FaceOrienter
from .dataclasses import Direction

def main():
    """Main function to run the face direction detection."""
    parser = argparse.ArgumentParser(description="Analyze face direction from an image file.")
    parser.add_argument("input_file", help="Path to the input image file")
    parser.add_argument("--show", action="store_true", help="Show the visualization on the image")
    args = parser.parse_args()

    face_orienter = FaceOrienter()
    yaw, pitch, roll = face_orienter.orient(args.input_file, args.show)

    if yaw is not None and pitch is not None:
        face_direction = Direction(yaw, pitch)
        print(f"Yaw: {face_direction.yaw:.2f} degrees, Pitch: {face_direction.pitch:.2f} degrees")
        print(f"Direction: {face_direction}")
    else:
        print("Could not determine face direction.")

if __name__ == "__main__":
    main()
