import sys
sys.path.insert(1, '../Phase1')
from features_images import FeaturesImages
from sklearn.cluster import KMeans
import os
import misc
from pathlib import Path
import numpy as np
from tqdm import tqdm

def get_sift_features():

    parent_directory_path = Path(os.path.dirname(__file__)).parent
    pickle_file_directory = os.path.join(parent_directory_path, 'Phase1')
    dataset_images_features = misc.load_from_pickle(pickle_file_directory, 'SIFT')

    input_k_means = []
    sum = 0
    images_num=0

    # To store the key_point descriptors in a 2-d matrix of size (k1+k2+k3...+kn)*128

    for image_id, feature_vector in dataset_images_features.items():
        for feature_descriptor in feature_vector:
            # Note : haven't used x,y,scale,orientation
            input_k_means.append(feature_descriptor[4:])
        sum = sum + len(feature_vector)
        images_num = images_num + 1
    n_clusters = int(sum / images_num)
    kmeans = KMeans(n_clusters)
    print('Applying k-means algorithm on all the keypoint descriptors  of all images')
    tqdm(kmeans.fit(input_k_means))

    row_s=0
    row_e=0
    k=0

    image_features = {}
    print('Equating the number of features for all the images : ')
    for image_id, feature_vector in tqdm(dataset_images_features.items()):
        row_s = row_s + k
        k = len(feature_vector)
        row_e = row_e + k
        closest_cluster = kmeans.predict(input_k_means[row_s:row_e])
        reduced_feature_img = [0] * n_clusters

        for cluster_num in closest_cluster:
            reduced_feature_img[cluster_num] = reduced_feature_img[cluster_num]+1;
        image_features[image_id] = reduced_feature_img


    folder_images_features_dict = {}
    for image_id, feature_vector in dataset_images_features.items():
        folder_images_features_dict[image_id] = image_features[image_id]

    print(len(image_features))
    reduced_pickle_file_folder = os.path.join(os.path.dirname(__file__), 'pickle_files')
    misc.save2pickle(folder_images_features_dict, reduced_pickle_file_folder, 'SIFT_NEW')