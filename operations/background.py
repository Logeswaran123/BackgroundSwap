import cv2
from diffusers import StableDiffusionPipeline
import numpy as np
import torch
from torch import autocast


HEIGHT = 512
WIDTH  = 512

class Background():
    def __init__(self, prompt, bg_img_path) -> None:
        self.prompt = prompt
        self.bg_img_path = bg_img_path
        self.use_sd = False

        if bg_img_path is None:
            if prompt:
                self.use_sd = True
            else:
                print("Background image path (--bg) or text prompt (--prompt) must be provided.")
                exit()

    def get_background(self):
        return self._generate_image() if self.use_sd else self._get_image()

    def _generate_image(self):
        access_token = self._get_access_token()

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=access_token,
                                                    torch_dtype=torch.float16, revision='fp16')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipe = pipe.to(device)
        print("\nstable-diffusion-v1-4 model loaded successfully")

        print(f"\nGenerating background image using prompt: {self.prompt}\n")
        with autocast("cuda"):
            image = pipe(prompt=self.prompt, height=HEIGHT, width=WIDTH).images[0]

        return np.array(image)

    def _get_image(self):
        return cv2.imread(self.bg_img_path)

    def _get_access_token(self):
        return input("\nEnter Hugging face user access token: ")