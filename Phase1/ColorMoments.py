import scipy as sp
import cv2
import misc
import os
import math
import collections
import numpy as np


def euclidean_distance(feature_list1, feature_list2):
    return (sum([(a - b) ** 2 for a, b in zip(feature_list1, feature_list2)])) ** 0.5


class ColorMoments:
    def __init__(self):
        self.model_name = 'CM'
        self.similarity_fn = euclidean_distance
        self.reverse_sort = False

    def get_moments_channel(self, channel):
        moment_mean = sp.mean(channel)
        moment_sd = sp.std(channel)
        moment_skew = sp.stats.skew(channel.flatten())
        return [moment_mean, moment_sd, moment_skew]

    def compute(self, image):
        c1, c2, c3 = cv2.split(image)
        image_features_y = self.get_moments_channel(c1)
        image_features_u = self.get_moments_channel(c2)
        image_features_v = self.get_moments_channel(c3)
        features_tuple = [feature for feature in zip(image_features_y, image_features_u, image_features_v)]
        combined_feature = []
        for feature in features_tuple:
            combined_feature.append(feature[0])
            combined_feature.append(feature[1])
            combined_feature.append(feature[2])
        hist, bins = np.histogram(combined_feature, bins=3)
        # print(hist)
        # return combined_feature
        return hist




