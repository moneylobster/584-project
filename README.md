# EE584 Term Project Code

## Requirements

Python 3 with pip libraries:

* OpenCV
* numpy
* matplotlib
* scikit-learn
* scikit-image

## Installation

* Clone/download the git repository:

`git clone https://github.com/moneylobster/584-project/tree/master`

* Install segment anything for python following the instructions in [this link](https://github.com/facebookresearch/segment-anything)

* Download the vit-h model weights from the same link, place in the `1. code` folder of the project.

## Usage

* Place images to evaluate in the `1. code/images/` folder
* Execute `1. code/main.py`.
* A window for each image will pop up. Hold down the mouse button and drag to draw a bounding box on the image, then press `ESC`.
* The results will be saved in the `1. code/results/` folder