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

    # get number of clusters
    cluster_num=int(input("Enter the number of clusters to segment into: "))

    plt.figure(figsize=(10,10))
    
    # segment anything
    print("Running Segment Anything...")
    seg_res=segmentanything.do_bbox(img, np.array(bbox))
    plt.subplot(2,2,1)
    plt.imshow(img)
    segmentanything.show_mask(seg_res[0], plt.gca())
    segmentanything.show_box(bbox, plt.gca())
    plt.axis('off')
    plt.title("sam")

    # k-means 5-D
    print("Running k-means (5D)...")
    kmean5_res=kMeans_meanShift.Kmeans5d_segment(img, cluster_num)
    plt.subplot(2,2,2)
    plt.imshow(kmean5_res)
    plt.axis('off')
    plt.title("kmeans 5d")

    # k-means 3-D
    print("Running k-means (3D)...")
    kmean3_res=kMeans_meanShift.Kmeans3d_segment(img, cluster_num)
    plt.subplot(2,2,3)
    plt.imshow(kmean3_res)
    plt.axis('off') 
    plt.title("kmeans 3d")
    

    plt.tight_layout()
    plt.show()


main("images/balls.png")
