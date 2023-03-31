import cv2


class Background():
    def __init__(self, prompt, bg_img_path) -> None:
        self.prompt = prompt
        self.bg_img_path = bg_img_path
        self.use_sd = False

        if bg_img_path == None:
            if prompt != None:
                self.use_sd = True
            else:
                print("Background image path (--bg) or text prompt must be provided (--prompt).")
                exit()

    def get_background(self):
        return self._generate_image() if self.use_sd else self._get_image()

    def _generate_image(self):
        pass

    def _get_image(self):
        return cv2.imread(self.bg_img_path)