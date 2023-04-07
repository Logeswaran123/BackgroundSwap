import argparse

from operations.foreground import Foreground
from operations.background import Background


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True, type=str,
                                        help="Path to input images directory.")
    parser.add_argument('-o', "--output", required=True, type=str,
                                        help="Path to output images directory.")
    parser.add_argument('-b', "--bg", required=False, type=str,
                                        help="Path to background image file.")
    parser.add_argument('-m', "--mode", required=False, default=0, type=int, choices=[0, 1, 2],
                                        help="Mode of segmentation to be used to generate mask. \
                                            {0:Mediapipe, 1:Deeplabv3, 2:Yolov8}")
    parser.add_argument('-p', "--prompt", required=False,
                                        help="Text prompt to generate background.")
    parser.add_argument('-n', "--nst", required=False, action='store_true', default=False,
                                        help="Perform Neural Style Transfer")
    return parser


def main():
    args = argparser().parse_args()
    background = Background(args.prompt, args.bg).get_background()
    foreground = Foreground(args.input, args.output, background, args.mode, args.nst)
    foreground.perform_segmentation()


if __name__ == "__main__":
    main()