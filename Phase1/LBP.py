from skimage import feature
import numpy as np
from scipy.stats import itemfreq
from itertools import zip_longest
import misc
import os
import collections


def euclidean_distance(dist1, dist2):
    return (sum([(a-b)**2 for a,b in zip(dist1, dist2)]))**0.5


def chisquare(dist1, dist2):
    return sum([((a-b)**2/(a+b)) if a+b != 0 else 0 for a,b in zip(dist1, dist2)])


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius
        self.similarity_fn = euclidean_distance
        self.reverse_sort = False

    def compute(self, image):
        return self.compute_lbp(image)

    def compute_lbp(self, image, eps=1e-7):
        lbp = feature.local_binary_pattern(image=image, P=self.numPoints, R=self.radius, method='uniform')
        # x = itemfreq(lbp.ravel())
        # hist = x[:, 1]/sum(x[:, 1])
        vecimgLBP = lbp.flatten()
        hist, hist_edges = np.histogram(vecimgLBP, bins=10)
        return hist



