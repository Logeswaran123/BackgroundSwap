import cv2
import numpy as np
import os

from operations.neural_style_transfer import NeuralStyleTransfer
import operations.utils as utils


YOLOV8_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt"


class Mediapipe():
    def __init__(self, input_dir, output_dir, background, neural_style_transfer) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.background = background
        self.neural_style_transfer = neural_style_transfer
        self.mp_drawing = None
        self.mp_selfie_segmentation = None

    def perform_segmentation(self, ):
        import mediapipe as mp

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)

        print("\nPerforming segmentation using Mediapipe...\n")
        if not os.listdir(self.input_dir):
            return
        utils.create_dir(self.output_dir)

        for idx, file in enumerate(os.listdir(self.input_dir)):
            file_path = self.input_dir + "/" + file
            print(f"\nFile {idx + 1} : {file_path}")

            image = cv2.imread(file_path)
            image_height, image_width, _ = image.shape

            results = self.mp_selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            condition = cv2.GaussianBlur(np.array(condition, dtype=np.float32), (5, 5), 11)

            bg_image = NeuralStyleTransfer().perform(image, self.background) if self.neural_style_transfer else self.background
            bg_image = cv2.GaussianBlur(cv2.resize(bg_image, image.shape[:2][::-1]), (55, 55), 0)

            output_image = np.where(condition, image, bg_image)
            cv2.imwrite(self.output_dir + "/" + str(idx) + '.png', output_image)


class Deeplabv3():
    def __init__(self, input_dir, output_dir, background, neural_style_transfer) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.background = background
        self.neural_style_transfer = neural_style_transfer

    def perform_segmentation(self, ):
        import torch
        from torchvision import transforms

        print("\nPerforming segmentation using Deeplabv3...\n")
        if not os.listdir(self.input_dir):
            return
        utils.create_dir(self.output_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
        model.to(device).eval()

        for idx, file in enumerate(os.listdir(self.input_dir)):
            file_path = self.input_dir + "/" + file
            print(f"\nFile {idx + 1} : {file_path}")

            image = cv2.imread(file_path)
            image_height, image_width, _ = image.shape

            input_tensor = transforms.ToTensor()(image)
            input_tensor = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)['out'][0]

            output_array = output.argmax(0).cpu().numpy().astype(np.uint8)
            condition = np.stack((output_array,) * 3, axis=-1) > 0.1
            condition[output_array == 15] = 255
            condition = cv2.GaussianBlur(np.array(condition, dtype=np.float32), (5, 5), 11)

            bg_image = NeuralStyleTransfer().perform(image, self.background) if self.neural_style_transfer else self.background
            bg_image = cv2.GaussianBlur(cv2.resize(bg_image, image.shape[:2][::-1]), (55, 55), 0)

            output_image = np.where(condition, image, bg_image)
            cv2.imwrite(self.output_dir + "/" + str(idx) + '.png', output_image)


class YOLOv8():
    def __init__(self, input_dir, output_dir, background, neural_style_transfer) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.background = background
        self.neural_style_transfer = neural_style_transfer

    def perform_segmentation(self, ):
        from ultralytics import YOLO
        from urllib import request

        print("\nPerforming segmentation using Yolov8...\n")
        utils.create_dir(os.getcwd() + "/yolo-model")
        request.urlretrieve(YOLOV8_URL, os.getcwd() + "/yolo-model/yolov8x-seg.pt")
        model = YOLO('yolo-model/yolov8x-seg.pt')

        for idx, file in enumerate(os.listdir(self.input_dir)):
            file_path = self.input_dir + "/" + file
            print(f"\nFile {idx + 1} : {file_path}")

            image = cv2.imread(file_path)
            image_height, image_width, _ = image.shape

            output = model.predict(file_path)[0]
            output = output.masks.masks[0]

            output_array = output.cpu().numpy().astype(np.uint8) * 255
            condition = cv2.resize(output_array, image.shape[:2][::-1])
            condition = np.stack((condition,) * 3, axis=-1) > 0.1
            condition = cv2.GaussianBlur(np.array(condition, dtype=np.float32), (5, 5), 11)

            bg_image = NeuralStyleTransfer().perform(image, self.background) if self.neural_style_transfer else self.background
            bg_image = cv2.GaussianBlur(cv2.resize(bg_image, image.shape[:2][::-1]), (55, 55), 0)

            output_image = np.where(condition, image, bg_image)
            cv2.imwrite(self.output_dir + "/" + str(idx) + '.png', output_image)


class Foreground():
    def __init__(self, input_dir, output_dir, background, mode = 0, neural_style_transfer = False) -> None:
        kwargs = {"input_dir" : input_dir,
                  "output_dir" : output_dir,
                  "background" : background,
                  "neural_style_transfer" : neural_style_transfer}

        modes = {
            0: Mediapipe(**kwargs),
            1: Deeplabv3(**kwargs),
            2: YOLOv8(**kwargs),
        }

        self.instance = modes.get(mode)

    def perform_segmentation(self, ):
        self.instance.perform_segmentation()
