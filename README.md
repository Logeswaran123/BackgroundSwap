# [In Progress] The Traveller
An application that changes the background of a person using a text prompt describing the background.

## Description :scroll:
Generate the background of a person with a diffusion model.

## Directory Structure
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
```

## General Requirements :mage_man:
* Place all the content images in the input directory.
* Place the background image in background directory.
* If Neural Style Transfer is not set, the background image will be set as background of the content image.
* If Neural Style Transfer is set, the background image's style will be applied to the background of the content image.
* If blur is set, then the background of the content image will be blurred.

## Code Requirements :mage_woman:
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
|![](https://github.com/Logeswaran123/The-Traveller/blob/main/images/img_1.jpeg)||![](https://github.com/Logeswaran123/The-Traveller/blob/main/images/img_1_nst.png)|

| Input image | Prompt | Output image with background generated using prompt |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![](https://github.com/Logeswaran123/The-Traveller/blob/main/images/img_1.jpeg)| "((dim lit)), japanese alleyway with trees on the side, 8k, realistic, detailed, sharp, trending, incredible pixel details" |![](https://github.com/Logeswaran123/The-Traveller/blob/main/images/img_1_nonst.png)|

## References :page_facing_up:
* Neural Style Transfer (optimization method) | [Repository](https://github.com/gordicaleksa/pytorch-neural-style-transfer)
* Ultralytics YOLOv8 | [Documentation](https://docs.ultralytics.com/)
