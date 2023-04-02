import argparse

from operations.segmentation import Mask
from operations.image_generator import Background


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                                        help="Path to input images directory.", type=str)
    parser.add_argument('-o', "--output", required=True,
                                        help="Path to output images directory.", type=str)
    parser.add_argument('-b', "--bg", required=False,
                                        help="Path to background image file.", type=str)
    parser.add_argument('-m', "--mediapipe", required=False, action='store_true', default=False,
                                        help="Use Mediapipe if set to True, else use DeeplabV3 for mask generation.")
    parser.add_argument('-p', "--prompt", required=False, # change to True
                                        help="Text prompt to generate background.")
    return parser


def main():
    args = argparser().parse_args()
    background = Background(args.prompt, args.bg).get_background()
    mask = Mask(args.input, args.output, background, args.mediapipe)
    mask.perform_segmentation()


if __name__ == "__main__":
    main()