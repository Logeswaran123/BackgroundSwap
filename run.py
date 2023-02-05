import argparse
import cv2
from PIL import Image
from ultralytics import YOLO


class Segmenter:
    def __init__(self, model_path, source, **kwargs):
        self.model_path = model_path
        self.source = source
        self.overrides = {}
        self.overrides.update(kwargs)

    def predict(self):
        model = YOLO(self.model_path)
        results = model.predict(source=self.source, **self.overrides)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", required=True, help="Path to model", type=str)
    parser.add_argument('-s', "--source", required=False, default="0",
                        help="Source directory for images or videos", type=str)
    args = parser.parse_args()

    kwargs = {'task': 'segment', 'retina_masks': True, 'classes': 0, 'save': "./"}
    segmenter = Segmenter(args.model, args.source, **kwargs)
    segmenter.predict()