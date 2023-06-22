'''
get an image file and show segmentation results from multiple methods
'''

import matplotlib.pyplot as plt
import cv2

# models
import segmentanything
import bounding_box

import kMeans_meanShift
import thresholding

def main(imfile):
    """segment image with multiple methods

    Parameters
    ----------
    imfile : filepath
        path to image to segment
    """
    img=cv2.imread(imfile)
    
    # segment anything
    bbox=bounding_box.start(img)
    print(bbox)
