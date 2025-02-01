import argparse
from .face_orienter import FaceOrienter


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("input_file", help="Path to the input video file")
    parser.add_argument("--show", action="store_true", help="Show the project on the image")
    args = parser.parse_args()

    face = FaceOrienter()
    yaw, pitch, roll = face.orient(args.input_file, args.show)
    print(f"Yaw: {yaw:.2f} degrees, Pitch: {pitch:.2f} degrees, Roll: {roll:.2f} degrees")

if __name__ == "__main__":
    main()

