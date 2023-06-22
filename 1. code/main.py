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
    plot_h=3
    plot_w=2
    
    img=cv2.imread(imfile)

    # get a bounding box
    print("Hold down the mouse to select a box, then press ESC")
    bbox=bounding_box.start(img.copy())
    print(f"Bounding box coordinates: {bbox}")

    # get number of clusters
    cluster_num=int(input("Enter the number of clusters to segment into: "))

    plt.figure(figsize=(10,10))
    
    # original image
    plt.subplot(plot_h,plot_w,1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Original Image")


    # segment anything
    print("Running Segment Anything...")
    seg_res=segmentanything.do_bbox(img, np.array(bbox))
    plt.subplot(plot_h,plot_w,2)
    plt.imshow(img)
    segmentanything.show_mask(seg_res[0], plt.gca())
    segmentanything.show_box(bbox, plt.gca())
    plt.axis('off')
    plt.title("Segment Anything")

    # k-means 5-D
    print("Running k-means (5D)...")
    kmean5_res=kMeans_meanShift.Kmeans5d_segment(img, cluster_num)
    plt.subplot(plot_h,plot_w,3)
    plt.imshow(kmean5_res)
    plt.axis('off')
    plt.title("k-means (5D)")

    # k-means 3-D
    print("Running k-means (3D)...")
    kmean3_res=kMeans_meanShift.Kmeans3d_segment(img, cluster_num)
    plt.subplot(plot_h,plot_w,4)
    plt.imshow(kmean3_res)
    plt.axis('off') 
    plt.title("k-means (3D)")

    # otsu
    print("Running Otsu...")
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu_res, otsu_thresh=thresholding.otsu(img_gray)
    plt.subplot(plot_h,plot_w,5)
    plt.imshow(cv2.cvtColor(otsu_res.astype(np.uint8),cv2.COLOR_GRAY2RGB)*255)
    plt.axis('off')
    plt.title("Otsu")

    plt.tight_layout()
    plt.show()


main("images/balls.png")
