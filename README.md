# 📸 BackgroundSwap 🖼️
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Logeswaran123/The-Traveller/blob/main/LICENSE)<br/><br/>
Easily replace the background of a content image with a background image of user's choice. If user does not have a background image, BackgroundSwap includes a stable diffusion model that can generate a background based on the prompt provided by the user. BackgroundSwap also enables users to apply the style of the background image to the background of the content image.

## Description :scroll:
BackgroundSwap changes the background or style of the content image with background image. Three stages are used to realize the output image depending on the modes of usage. Firstly, the user provided background is taken, if not provided, then background image is generated using user provided prompt with Stable Diffusion model. Secondly, mask is created for the content image's background using Mediapipe, Deeplabv3 or YOLOv8 model. Thirdly, depending on user provided argument, the style of background image is applied to background of content image using VGG19 network by guiding the stylized pixels.

![](https://github.com/Logeswaran123/BackgroundSwap/blob/main/images/flow%20diagram.png)

### How does Neural Style Transfer work
When an image is passed through a CNN, the image is brokend down to it's feature representations by conv layers. These feature representations can be the edges, lines, or more noticeable semantics like eye, nose. The content representation from the content image can be extracted from these intermediate feature representation. Now, the style representation of the style image is obtained differently. Style, by definition, means the texture, color, or mood. Here, style representation can be obtained by combining the information of the intermediate feature representations from every conv layer of the CNN. This is achieved by gram matrix. A gram matrix is a dot product of feature representation and it's transpose.

Now, to progressively generate a better target image, we have to monitor a loss parameter. The loss parameter in Neural Style Transfer task is the sum of content loss and style loss. The content loss is the MSE of target and content representations. The style loss is the MSE of style representation of target and style representation of style image. The main idea is to lower the loss through gradient descent and backpropagation.

## Directory Structure 📁
```
.
├── input
      ├── content image 1
      ├── content image 2
      .
      .
      .
├── background
      ├── background image 1
├── operations
      ├── background.py
      ├── foreground.py
      ├── models.py
      ├── neural_style_transfer.py
      ├── utils.py
├── run.py
├── setup.py
├── requirements.txt
```

## General Requirements :mage_man:
* Place all the content images in the input directory.
* Place the background image in background directory.
* If Neural Style Transfer is not set, the background image will be set as background of the content image.
* If Neural Style Transfer is set, the background image's style will be applied to the background of the content image.
* If blur is set, then the background of the content image will be blurred.

## Code Requirements :mage_woman:
Use Python 3.8.13. Setup conda environment, git clone repo and run the below command,
```python3
python setup.py
```

## How to run :running_man:
```python3
python run.py --input <dir path to input images> --output <dir path to store output images> --bg <dir path to background image> --mode <mode of segmentation>
```
<b>Arguments:</b><br/>
| Argument | Description | Value |
|:--------:|:-----------:|:-----:|
| --input  | Path to input images directory | String |
| --output | Path to output images directory | String |
| --bg | Path to background image file | String |
| --mode | Mode of segmentation to be used to generate mask | Integer<br/>0 : Mediapipe, 1 : Deeplabv3, 2 : Yolov8 |
| --prompt | Text prompt to generate background. Used when --bg is not provided | String |
| --nst | Perform Neural Style Transfer of background | store_true |
| --blur | Perform Gaussian Blur to Background | store_true |

## Results :bar_chart:
| Input image | Background image | Output image with Neural Style Transfer |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![](https://github.com/Logeswaran123/The-Traveller/blob/main/images/img_1.jpeg)|![](https://github.com/Logeswaran123/The-Traveller/blob/main/images/bg_1.jpg)|![](https://github.com/Logeswaran123/The-Traveller/blob/main/images/img_1_nst.png)|

| Input image | Prompt | Output image with background generated using prompt |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![](https://github.com/Logeswaran123/The-Traveller/blob/main/images/img_1.jpeg)| "((dim lit)), japanese alleyway with trees on the side, 8k, realistic, detailed, sharp, trending, incredible pixel details" |![](https://github.com/Logeswaran123/The-Traveller/blob/main/images/img_1_nonst.png)|

## References :page_facing_up:
* Neural Style Transfer (optimization method) | [Repository](https://github.com/gordicaleksa/pytorch-neural-style-transfer)
* Basic Intuition And Guide to Neural Style Transfer | [Article](https://pub.towardsai.net/basic-intuition-on-neural-style-transfer-idea-c5ac179d1530)
* Ultralytics YOLOv8 | [Documentation](https://docs.ultralytics.com/)

## Contributions 👩‍💻
Contributions to BackgroundSwap are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request. Let's work together to make BackgroundSwap even better!

Happy Learning! 😄
