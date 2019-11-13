from pathlib import Path
import misc
import os
from tqdm import tqdm
import LBP
import HOG
import ColorMoments
import SIFT
import sys
from sklearn.cluster import MiniBatchKMeans
import numpy as np

'''
Class for handling all the feature vector generation for both single and multiple images
'''


class FeaturesImages:

    # Initialize the class variables model name, folder path and model
    def __init__(self, model_name, folder_path=None):
        self.model_name = model_name
        self.folder_path = folder_path
        self.split_windows = False
        self.model = None
        if self.model_name == 'LBP':
            self.model = LBP.LocalBinaryPatterns(8, 1)
            self.split_windows = True
        elif self.model_name == 'HOG':
            self.model = HOG.Hog(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        elif self.model_name == 'CM':
            self.model = ColorMoments.ColorMoments()
            self.split_windows = True
        elif self.model_name == 'SIFT':
            self.model = SIFT.SIFT()

    '''
    A getter function to get the initialised feature extraction model.
    '''
    def get_model(self):
        return self.model

    '''
    For the folder path specified, the below function computes the feature vectors
    and stores them in a pickle file based on the respective feature extraction model using
    a helper function 'compute_image_features'. Use of the package 'tqdm' shows a 
    smart progress meter.
    '''

    def compute_features_images_folder(self):
        if self.model is None:
            raise Exception("No model is defined")
        else:
            folder = os.path.join(Path(os.path.dirname(__file__)).parent, self.folder_path)
            files_in_directory = misc.get_images_in_directory(folder)
            features_image_folder = []
            for file, path in tqdm(files_in_directory.items()):
                image_feature = self.compute_image_features(path, print_arr=False)
                features_image_folder.append(image_feature)
            images = list(files_in_directory.keys())
            folder_images_features_dict = {}
            for i in range(len(images)):
                folder_images_features_dict[images[i]] = features_image_folder[i]

            # print(folder_images_features_dict)

            if self.model_name == 'SIFT':
                misc.save2pickle(folder_images_features_dict, os.path.dirname(__file__),
                                 feature=self.model_name + "_OLD")
                folder_images_features_dict_sift_new = self.compute_sift_new_features(folder_images_features_dict)
                misc.save2pickle(folder_images_features_dict_sift_new, os.path.dirname(__file__),
                                 feature=self.model_name)
            else:
                misc.save2pickle(folder_images_features_dict, os.path.dirname(__file__), feature=self.model_name)


    '''
    Given an image path, based on the model requirements the features vectors are retrieved and
    based on the attribute value for 'print_arr' the function either prints the feature vector 
    or returns the feature vector.
    '''

    def compute_image_features(self, image, print_arr=False):
        image_feature = []
        try:
            image_path = os.path.join(os.path.dirname(__file__), image)
            image = misc.read_image(image_path)
            converted_image = misc.convert2gray(image)
            if self.model_name == 'CM':
                converted_image = misc.convert2yuv(image)
            if self.model_name == 'HOG':
                converted_image = misc.resize_image(converted_image, (120, 160))
            if self.split_windows:
                windows = misc.split_into_windows(converted_image, 100, 100)
                for window in windows:
                    window_pattern = self.model.compute(window)
                    if len(image_feature) == 0:
                        image_feature = window_pattern
                    else:
                        image_feature = np.concatenate([image_feature, window_pattern])
            else:
                image_feature = self.model.compute(converted_image)
        except OSError as e:
            print("Features_image", e.strerror)
            sys.exit()
        finally:
            if not print_arr:
                return image_feature
            else:
                print(image_feature)

    def compute_sift_new_features(self, dataset_images_features):
        input_k_means = []
        sum = 0
        images_num = 0

        # To store the key_point descriptors in a 2-d matrix of size (k1+k2+k3...+kn)*128
        min_val = 50
        for image_id, feature_vector in dataset_images_features.items():
            for feature_descriptor in feature_vector:
                # Note : haven't used x,y,scale,orientation
                input_k_means.append(feature_descriptor[4:])
            sum = sum + len(feature_vector)
            len_featurevec = len(feature_vector)
            if  len_featurevec!=1 and len_featurevec < min_val:
                min_val = len(feature_vector)
            images_num = images_num + 1

        # n_clusters = int(sum / images_num)
        n_clusters = 70
        # int(sum / images_num) #taking so much time - better to fix some value
        kmeans = MiniBatchKMeans(n_clusters, random_state=42)
        kmeans.fit(input_k_means)

        row_s = 0
        row_e = 0
        k = 0

        image_features = {}
        for image_id, feature_vector in dataset_images_features.items():
            row_s = row_s + k
            k = len(feature_vector)
            row_e = row_e + k
            closest_cluster = kmeans.predict(input_k_means[row_s:row_e])
            reduced_feature_img = [0] * n_clusters

            for cluster_num in closest_cluster:
                reduced_feature_img[cluster_num] = reduced_feature_img[cluster_num] + 1
            image_features[image_id] = reduced_feature_img

        folder_images_features_dict = {}
        for image_id, feature_vector in dataset_images_features.items():
            folder_images_features_dict[image_id] = image_features[image_id]

        return folder_images_features_dict
