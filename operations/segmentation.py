import cv2
import mediapipe as mp
import numpy as np

from .utils import *


class Mediapipe():
    def __init__(self, input_dir, output_dir, background) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.background = background
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

    def perform_segmentation(self, ):
        if not os.listdir(self.input_dir):
            return
        create_dir(self.output_dir)

        for idx, file in enumerate(os.listdir(self.input_dir)):
            file_path = self.input_dir + "/" + file
            print(f"\nFile {idx + 1} : {file_path}")

            image = cv2.imread(file_path)
            image_height, image_width, _ = image.shape

            results = self.mp_selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

            bg_image = cv2.GaussianBlur(cv2.resize(self.background, image.shape[:2][::-1]), (55, 55), 0)

            output_image = np.where(condition, image, bg_image)
            cv2.imwrite(self.output_dir + "/" + str(idx) + '.png', output_image)


class DeeplabV3():
    def __init__(self, input_dir, output_dir, background) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.background = background

    def perform_segmentation(self, ):
        pass


class Mask():
    def __init__(self, input_dir, output_dir, background, use_mediapipe = True) -> None:
        kwargs = {"input_dir":input_dir, "output_dir":output_dir, "background":background}
        self.instance = Mediapipe(**kwargs) if use_mediapipe else DeeplabV3(**kwargs)

    def perform_segmentation(self, ):
        self.instance.perform_segmentation()
