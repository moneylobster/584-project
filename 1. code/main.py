'''
get an image file and show segmentation results from multiple methods
'''

import matplotlib.pyplot as plt
import cv2
import numpy as np

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

    # get a bounding box
    print("Hold down the mouse to select a box, then press ESC")
    bbox=bounding_box.start(img.copy())
    print(f"Bounding box coordinates: {bbox}")
    
    # segment anything
    print("Running Segment Anything...")
    seg_res=segmentanything.do_bbox(img, np.array(bbox))
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    segmentanything.show_mask(seg_res[0], plt.gca())
    segmentanything.show_box(bbox, plt.gca())
    plt.axis('off')
    plt.show()
