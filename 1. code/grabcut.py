import numpy as np
import cv2 
from matplotlib import pyplot as plt

def grabcut_segment(img, bbox):
    """segment image with grabcut

    Parameters
    ----------
    img : numpy.ndarray
        image to segment
    bbox : numpy.ndarray
        bounding box coordinates

    Returns
    -------
    numpy.ndarray
        segmented image
    """
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    rect = (bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1])
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    
    img = img*mask2[:,:,np.newaxis]
    
    return img