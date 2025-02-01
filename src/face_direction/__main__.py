import argparse
from .face_orienter import FaceOrienter


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file.")
    parser.add_argument("input_file", help="Path to the input video file")
    args = parser.parse_args()

    face = FaceOrienter()
    face.orient(args.input_file)

if __name__ == "__main__":
    main()

