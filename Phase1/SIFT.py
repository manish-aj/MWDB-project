#!/usr/bin/python
import cv2
import os
import collections
import misc
import sys
from tqdm import tqdm


# def euclidean_distance(image1, image2):
#     m = len(image1)
#     n = len(image2)
#     '''
#         In the image feature list, in each vector, the first 4 are x,y,scale and orientation
#         so the descriptor will be vector[4:] which is of length 128.
#     '''
#     res = 0
#     for i in range(m):
#         min_dist = sys.maxsize
#         for j in range(n):
#             euclid_dist = sum([(a-b)**2 for a, b in zip(image1[i][4:], image2[j][4:])])**0.5
#             if min_dist > euclid_dist:
#                 min_dist = euclid_dist
#         res += min_dist
#     return res/m

def euclidean_distance(dist1, dist2):
    return (sum([(a-b)**2 for a, b in zip(dist1, dist2)]))**0.5


class SIFT:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.similarity_fn = euclidean_distance
        self.reverse_sort = False

    def compute(self, image):
        keypoints, descriptor = self.sift.detectAndCompute(image, None)
        number_of_keypoints = len(keypoints)
        image_feature = []
        for index in range(number_of_keypoints):
            keypoint_vector = [keypoints[index].pt[0], keypoints[index].pt[1], keypoints[index].size,
                               keypoints[index].angle]
            keypoint_vector += list(descriptor[index])
            image_feature.append(keypoint_vector)


        return image_feature
