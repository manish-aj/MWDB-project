from skimage import feature
import misc
import os
import collections
import numpy as np


def cosine_similarity(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    sum_xy = sum([(a-mean_x) * (b-mean_y) for a, b in zip(x, y)])
    sum_square_x = sum([(a-mean_x) * (a-mean_x) for a in x])
    sum_square_y = sum([(b-mean_y) * (b-mean_y) for b in y])
    cosine_sim = sum_xy / (pow(sum_square_x, 0.5) * pow(sum_square_y, 0.5))
    return cosine_sim


class Hog:
    def __init__(self, orientations, pixels_per_cell, cells_per_block):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.similarity_fn = cosine_similarity
        self.reverse_sort = True

    def compute(self, image):
        return self.compute_hog(image)

    def compute_hog(self, image):
        (H, hogImage) = feature.hog(image, orientations=self.orientations,
                                    pixels_per_cell=self.pixels_per_cell,
                                    cells_per_block=self.cells_per_block, visualize=True)
        # misc.plot_image(hogImage)
        return H


