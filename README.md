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
** TODO **

## How to run :running_man:
** TODO **

## Results :bar_chart:
** TODO **

## References :page_facing_up:
** TODO **
