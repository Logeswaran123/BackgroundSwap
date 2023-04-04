import argparse

from operations.segmentation import Mask
from operations.image_generator import Background


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True, type=str,
                                        help="Path to input images directory.")
    parser.add_argument('-o', "--output", required=True, type=str,
                                        help="Path to output images directory.")
    parser.add_argument('-b', "--bg", required=False, type=str,
                                        help="Path to background image file.")
    parser.add_argument('-m', "--mode", required=False, default=0, type=int, choices=[0, 1, 2],
                                        help="Mode of segmentation to be used to generate mask. {0:Mediapipe, 1:Deeplabv3, 2:Yolov8}")
    parser.add_argument('-p', "--prompt", required=False, # change to True
                                        help="Text prompt to generate background.")
    return parser


def main():
    args = argparser().parse_args()
    background = Background(args.prompt, args.bg).get_background()
    mask = Mask(args.input, args.output, background, args.mode)
    mask.perform_segmentation()


if __name__ == "__main__":
    main()