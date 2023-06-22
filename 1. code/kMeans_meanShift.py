import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.preprocessing import MinMaxScaler


def normalization(img_orig):
    img = img_orig.copy()
    vectors = [img[i, j, :].tolist() + [i, j] for i in range(img.shape[0]) for j in range(img.shape[1])]
    vectors = np.array(vectors)  
    normalized = MinMaxScaler(feature_range=(0,1)).fit_transform(vectors)
    return normalized
def Kmeans3d_segment(img_orig, cluster_num):
    img = img_orig.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    normalized = normalization(img)
    normalized = normalized[:, :3]
    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init='auto').fit(normalized)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # find the color of each cluster
    cluster_color = np.zeros((cluster_num, 3))
    for i in range(cluster_num):
        cluster_color[i, :] = np.mean(normalized[labels==i, :3], axis=0)
    # create the segmented image
    img_seg = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_seg[i, j, :] = cluster_color[labels[i*img.shape[1]+j], :]
    return img_seg*255


def Kmeans5d_segment(img_orig,cluster_num):
    img = img_orig.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    normalized = normalization(img)
    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init='auto').fit(normalized)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # find the color of each cluster
    cluster_color = np.zeros((cluster_num, 3))
    for i in range(cluster_num):
        cluster_color[i, :] = np.mean(normalized[labels==i, :3], axis=0)
    # create the segmented image
    img_seg = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_seg[i, j, :] = cluster_color[labels[i*img.shape[1]+j], :]
    return img_seg*255


def MeanShift_segment(img_orig):
    img = img_orig.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    normalized = normalization(img)
    bandwidth = estimate_bandwidth(normalized, quantile=.04,n_samples=3000, n_jobs=-1)
    ms_1 = MeanShift(bandwidth = bandwidth , n_jobs=-1, bin_seeding=True, cluster_all=True).fit(normalized)
    labels = ms_1.labels_
    cluster_centers = ms_1.cluster_centers_
    print(labels)
    print(max(labels))
    cluster_color = np.zeros((max(labels)+1, 3))
    for i in range(max(labels)+1):
        cluster_color[i, :] = np.mean(normalized[labels==i, :3], axis=0)


    # create the segmented image
    img_seg = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_seg[i, j, :] = cluster_color[labels[i*img.shape[1]+j], :]
            
    return img_seg*255
