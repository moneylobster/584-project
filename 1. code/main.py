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
    
    # segment anything
    print("Running Segment Anything...")
    seg_res=segmentanything.do_bbox(img, np.array(bbox))
    # plt.figure(figsize=(10, 10))
    plt.subplots()
    plt.imshow(img)
    segmentanything.show_mask(seg_res[0], plt.gca())
    segmentanything.show_box(bbox, plt.gca())
    plt.axis('off')
    plt.show()

    # k-means 5-D
    print("Running k-means (5D)...")
    kmean_res=kMeans_meanShift.Kmeans_segment(img, cluster_num)
    # plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(kmean_res.astype(np.uint8), cv2.COLOR_HSV2RGB))
    plt.axis('off')
    plt.show()

    # k-means 3-D
    # print("Running k-means (3D)...")
    # kmean_res=kMeans_meanShift.Kmeans3d_segment(img, cluster_num)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(cv2.cvtColor(kmean_res.astype(np.uint8), cv2.COLOR_HSV2RGB))
    # plt.axis('off')
    # plt.show()


main("images/balls.png")
