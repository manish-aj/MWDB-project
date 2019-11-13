import sys
sys.path.insert(1, '../Phase1')
import misc
import os
import numpy as np
import pandas as pd
from pathlib import Path
from features_images import FeaturesImages
from prettytable import PrettyTable
from sklearn.cluster import MiniBatchKMeans
from PCA import PCAModel
from tqdm import tqdm



def euclidean_distance(dist1, dist2):
    return (sum([(a-b)**2 for a, b in zip(dist1, dist2)]))**0.5


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def reduce_subject_dim(subject_map):
    database_matrix = []

    for row_num, row in subject_map.items():
        database_matrix.append(row)

    # TO-DO CHANGE THE VALUE OF MAX_LATENT
    pcaobj = PCAModel(database_matrix, 10, '')
    pcaobj.decompose()
    return pcaobj.get_decomposed_data_matrix()


class Metadata:
    def __init__(self, test_images_list=None):
        self.test_images_list = test_images_list
        self.metadata_file_path = os.path.join(Path(os.path.dirname(__file__)).parent, 'data/HandInfo.csv')
        self.reduced_dimension_pickle_path = os.path.join(Path(os.path.dirname(__file__)).parent,
                                                          'Phase2', 'pickle_files')
        self.unlabeled_image_features = None
        self.images_metadata = None
        self.metadata_images_features = None
        self.set_images_metadata()

    def get_images_metadata(self):
        return self.images_metadata

    def set_images_metadata(self):
        self.images_metadata = pd.read_csv(self.metadata_file_path)

    def get_specific_metadata_images_list(self, feature_dict=None):

        if self.images_metadata is None:
            self.set_images_metadata()

        filtered_images_metadata = self.images_metadata

        # print('test_images_list', self.test_images_list)

        if self.test_images_list is not None:
            filtered_images_metadata = filtered_images_metadata[
                (filtered_images_metadata['imageName'].isin(self.test_images_list))]

        if feature_dict is not None:

            aspect_of_hand = feature_dict.get('aspectOfHand')

            accessories = feature_dict.get('accessories')

            gender = feature_dict.get('gender')

            if aspect_of_hand:
                filtered_images_metadata = filtered_images_metadata[
                    (filtered_images_metadata['aspectOfHand'].str.contains(aspect_of_hand))]

            if accessories:
                filtered_images_metadata = filtered_images_metadata[
                    (filtered_images_metadata['accessories'] == accessories)]

            if gender:
                filtered_images_metadata = filtered_images_metadata[
                    (filtered_images_metadata['gender']) == gender]

        images_list = filtered_images_metadata['imageName'].tolist()
        return images_list

    def sub_sub_list(self, sub1):
        if self.images_metadata is None:
            self.set_images_metadata()

        filtered_images_metadata = self.images_metadata
        if self.test_images_list is not None:
            filtered_images_metadata = filtered_images_metadata[
                (filtered_images_metadata['imageName'].isin(self.test_images_list))]

        subject_map = {}
        sub_ids_list = filtered_images_metadata['id'].unique().tolist()

        for sub_id in sub_ids_list:
            is_subject_id = filtered_images_metadata['id'] == sub_id
            subject_map[sub_id] = filtered_images_metadata[is_subject_id]

        parent_directory_path = Path(os.path.dirname(__file__)).parent
        pickle_file_directory = os.path.join(parent_directory_path, 'Phase1')
        dataset_images_features = misc.load_from_pickle(pickle_file_directory, 'SIFT_OLD')

        similarity_list_of_pair = [0]
        if sub1 not in sub_ids_list:
            return [tuple([-1, -1])]
        for sub2 in tqdm(sub_ids_list):
            if sub1!=sub2:
                sub_sub_val = self.subject_subject_similarity(subject_map[sub1], subject_map[sub2],
                                                          dataset_images_features, is_single_subject=True)
                similarity_list_of_pair.append(tuple([sub2, sub_sub_val]))

        similarity_list_of_pair = similarity_list_of_pair[1:]
        similarity_list_of_pair = sorted(similarity_list_of_pair, key=lambda x: x[1])

        return similarity_list_of_pair

    def subject_matrix(self):
        if self.images_metadata is None:
            self.set_images_metadata()

        filtered_images_metadata = self.images_metadata
        if self.test_images_list is not None:
            filtered_images_metadata = filtered_images_metadata[
                (filtered_images_metadata['imageName'].isin(self.test_images_list))]

        subject_map = {}
        sub_ids_list = filtered_images_metadata['id'].unique().tolist()
        sub_ids_list.sort() #just to look in sorted order OF SUBJECT IDS IN THE MATRIX
        #print(sub_ids_list)
        for sub_id in sub_ids_list:
            is_subject_id = filtered_images_metadata['id'] == sub_id
            subject_map[sub_id] = filtered_images_metadata[is_subject_id]

        # for now taking - number of latent semantics as 20(max_val)
        parent_directory_path = Path(os.path.dirname(__file__)).parent
        pickle_file_directory = os.path.join(parent_directory_path, 'Phase1')
        dataset_images_features = misc.load_from_pickle(pickle_file_directory, 'SIFT_OLD')

        similarity_matrix = []

        for sub1 in tqdm(sub_ids_list):
            similarity_row = []
            similarity_row_pair = [0]
            for sub2 in sub_ids_list:
                if sub1 == sub2:
                    similarity_row = similarity_row + [0]
                else:
                    sub_sub_val = self.subject_subject_similarity(subject_map[sub1],subject_map[sub2], dataset_images_features)
                    # print(sub_sub_val)
                    similarity_row = similarity_row + [sub_sub_val]

            similarity_matrix.append(similarity_row)

        p = PrettyTable()
        p.add_row(['SUBJECT/SUBJECT'] + sub_ids_list)
        i=0
        for row in similarity_matrix:
            row = [sub_ids_list[i]] + row
            p.add_row(row)
            i = i+1
        print(p.get_string(header=False, border=False))

        return similarity_matrix

    def subject_subject_similarity(self,data_frame1,data_frame2, dataset_images_features, is_single_subject=False):

        similarity_val = 0

        subject1_map = self.sub_16_map(data_frame1, dataset_images_features)
        subject2_map = self.sub_16_map(data_frame2, dataset_images_features)
        if is_single_subject:
            subject1_db_matrix = reduce_subject_dim(subject1_map)
            subject2_db_matrix = reduce_subject_dim(subject2_map)
        else:
            subject1_db_matrix = []
            subject2_db_matrix = []
            for key, value in subject1_map.items():
                subject1_db_matrix.append(value)

            for key, value in subject2_map.items():
                subject2_db_matrix.append(value)

        # print(subject1_db_matrix)
        # print(subject2_db_matrix)
        for i in range(16):
            similarity_val += euclidean_distance(subject1_db_matrix[i], subject2_db_matrix[i])

        return round(similarity_val, 3)

    def get_binary_image_metadata(self):

        if self.images_metadata is None:
            self.set_images_metadata()

        filtered_images_metadata = self.images_metadata

        # print('test_images_list', self.test_images_list)

        if self.test_images_list is not None:
            filtered_images_metadata = filtered_images_metadata[
                (filtered_images_metadata['imageName'].isin(self.test_images_list))]

        # print(filtered_images_metadata)
        image_binary_map ={}
        binary_image_metadata_matrix = []
        k=1
        # Matrix columns are left, right, dorsal, palmar, accessories, without accessories, male, female

        for row in filtered_images_metadata.itertuples():
            binary_matrix_row = []
            binary_matrix_row += [1 if 'left' in row.aspectOfHand else 0]
            binary_matrix_row += [1 if 'right' in row.aspectOfHand else 0]
            binary_matrix_row += [1 if 'dorsal' in row.aspectOfHand else 0]
            binary_matrix_row += [1 if 'palmar' in row.aspectOfHand else 0]
            binary_matrix_row += [row.accessories]
            binary_matrix_row += [1 if 'male' in row.gender else 0]
            binary_matrix_row += [1 if 'female' in row.gender else 0]

            binary_image_metadata_matrix.append(binary_matrix_row)

        return binary_image_metadata_matrix

    def set_metadata_image_features(self, pickle_file_path):
        self.metadata_images_features = misc.load_from_pickle(self.reduced_dimension_pickle_path, pickle_file_path)

    def set_unlabeled_image_features(self, model, test_image_id, decomposition):
        parent_directory_path = Path(os.path.dirname(__file__)).parent
        pickle_file_directory_phase1 = os.path.join(parent_directory_path, 'Phase1')
        test_image_features = list()
        test_image_features.append(misc.load_from_pickle(pickle_file_directory_phase1, model)[test_image_id])
        self.unlabeled_image_features = decomposition.decomposition_model.get_new_image_features_in_latent_space(
            test_image_features)

    def get_binary_label(self, feature_name):
        class_1_images_features = []
        class_0_images_features = []
        count = 0
        similarity_map = []
        if "Left" in feature_name:
            class_1_images = self.get_specific_metadata_images_list({'aspectOfHand': 'left'})
            class_0_images = self.get_specific_metadata_images_list({'aspectOfHand': 'right'})
            class_1_name = "left"
            class_0_name = "right"
            for image in class_1_images:
                class_1_images_features.append(self.metadata_images_features[image])
            for image in class_0_images:
                class_0_images_features.append(self.metadata_images_features[image])

        elif "Dorsal" in feature_name:
            class_1_images = self.get_specific_metadata_images_list({'aspectOfHand': 'dorsal'})
            class_0_images = self.get_specific_metadata_images_list({'aspectOfHand': 'palmar'})
            class_1_name = "dorsal"
            class_0_name = "palmar"
            for image in class_1_images:
                class_1_images_features.append(self.metadata_images_features[image])
            for image in class_0_images:
                class_0_images_features.append(self.metadata_images_features[image])

        elif "Gender" in feature_name:
            class_1_images = self.get_specific_metadata_images_list({'gender': 'male'})
            class_0_images = self.get_specific_metadata_images_list({'gender': 'female'})
            class_1_name = "male"
            class_0_name = "female"
            for image in class_1_images:
                class_1_images_features.append(self.metadata_images_features[image])
            for image in class_0_images:
                class_0_images_features.append(self.metadata_images_features[image])

        elif "Accessories" in feature_name:
            class_1_images = self.get_specific_metadata_images_list({'accessories': 1})
            class_0_images = self.get_specific_metadata_images_list({'accessories': 0})
            class_1_name = "with accessories"
            class_0_name = "without accessories"
            for image in class_1_images:
                class_1_images_features.append(self.metadata_images_features[image])
            for image in class_0_images:
                class_0_images_features.append(self.metadata_images_features[image])

        for metadata_feature_class_1 in class_1_images_features:
            similarity_map.append(tuple(("1", euclidean_distance(self.unlabeled_image_features,
                                                                 metadata_feature_class_1))))

        for metadata_feature_class_0 in class_0_images_features:
            similarity_map.append(tuple(("0", euclidean_distance(self.unlabeled_image_features,
                                                                 metadata_feature_class_0))))

        similarity_map = sorted(similarity_map, key=lambda x: x[1], reverse=False)
        print(feature_name, similarity_map)
        print("****************************************************************")

        count_1 = 0
        count_0 = 0

        k_nearest = len(self.metadata_images_features.items())//2
        for idx in range(k_nearest):
            if similarity_map[idx][0] == "1":
                count_1 += 1
            else:
                count_0 += 1

        if count_1 >= count_0:
            return class_1_name

        return class_0_name

    def sub_16_map(self, data_frame, dataset_images_features):
        metadata_arr_list = list()

        metadata_arr_list.append(['dorsal left', 1, 'male'])
        metadata_arr_list.append(['dorsal left', 0, 'male'])
        metadata_arr_list.append(['dorsal right', 1, 'male'])
        metadata_arr_list.append(['dorsal right', 0, 'male'])
        metadata_arr_list.append(['palmar left', 1, 'male'])
        metadata_arr_list.append(['palmar left', 0, 'male'])
        metadata_arr_list.append(['palmar right', 1, 'male'])
        metadata_arr_list.append(['palmar right', 0, 'male'])
        metadata_arr_list.append(['dorsal left', 1, 'female'])
        metadata_arr_list.append(['dorsal left', 0, 'female'])
        metadata_arr_list.append(['dorsal right', 1, 'female'])
        metadata_arr_list.append(['dorsal right', 0, 'female'])
        metadata_arr_list.append(['palmar left', 1, 'female'])
        metadata_arr_list.append(['palmar left', 0, 'female'])
        metadata_arr_list.append(['palmar right', 1, 'female'])
        metadata_arr_list.append(['palmar right', 0, 'female'])

        features_image = FeaturesImages('SIFT')
        metadata_vectors_16_map = {}
        count = 0
        for metadata_arr in metadata_arr_list:
            count = count + 1
            sift_cluster_vector = self.get_metadata_sift_feature_vector(data_frame, metadata_arr, dataset_images_features)
            metadata_vectors_16_map['combination'+str(count)] = sift_cluster_vector

        metadata_vectors_16_map = features_image.compute_sift_new_features(metadata_vectors_16_map)

        return metadata_vectors_16_map

    def get_metadata_sift_feature_vector(self, data_frame, metadata, dataset_images_features):

        input_k_means = []
        total = 0
        images_num = 0
        filtered_data_frame = data_frame[data_frame['aspectOfHand'].str.contains(metadata[0])]
        filtered_data_frame = filtered_data_frame[filtered_data_frame['accessories'] == metadata[1]]
        filtered_data_frame = filtered_data_frame[filtered_data_frame['gender'].str.contains(metadata[2])]
        list_filtered_images = filtered_data_frame['imageName'].tolist()

        # To store the key_point descriptors in a 2-d matrix of size (k1+k2+k3...+kn)*128

        if len(list_filtered_images) == 0:
            one_d = [0] * 128
            one_keypoint = [one_d]
            return one_keypoint

        min_val = 50
        for image_id, feature_vector in dataset_images_features.items():
            if image_id in list_filtered_images:
                for feature_descriptor in feature_vector:
                    # Note : haven't used x,y,scale,orientation
                    input_k_means.append(feature_descriptor[4:])
                if len(feature_vector) < min_val:
                    min_val = len(feature_vector)
                total = total + len(feature_vector)
                images_num = images_num + 1
        n_clusters = min(total, 70)

        if n_clusters != 0:
            kmeans = MiniBatchKMeans(n_clusters, random_state=42, max_iter=10)
            kmeans.fit(input_k_means)
        else:
            one_keypoint = [[0] * 128]
            return one_keypoint
        return kmeans.cluster_centers_

    def plot_subjects(self, main_subject, sub_sub_list, test_image_directory_path):
        sub_ids_list = [main_subject]
        sub_sub_similarity_pairs = {}
        for subject_pair in sub_sub_list:
            sub_ids_list += [subject_pair[0]]
            sub_sub_similarity_pairs[subject_pair[0]] = subject_pair[1]

        if self.images_metadata is None:
            self.set_images_metadata()

        filtered_images_metadata = self.images_metadata
        if self.test_images_list is not None:
            filtered_images_metadata = filtered_images_metadata[
                (filtered_images_metadata['imageName'].isin(self.test_images_list))]

        subject_map = {}
        for sub_id in sub_ids_list:
            is_subject_id = filtered_images_metadata['id'] == sub_id
            subject_map[sub_id] = filtered_images_metadata[is_subject_id]

        subject_images_list = {}
        count = 0
        for sub_id, data_frame in subject_map.items():
            if sub_id == main_subject:
                subject_images_list[sub_id] = {'imageList': [os.path.join(test_image_directory_path, image)
                                                             for image in data_frame['imageName'].tolist()],
                                               'value': 0}
            else:
                subject_images_list[sub_id] = {'imageList': [os.path.join(test_image_directory_path, image)
                                                             for image in data_frame['imageName'].tolist()],
                                               'value': sub_sub_similarity_pairs[sub_id]}
            count += 1
            if count > 3:
                break

        misc.plot_similar_images(subject_images_list, subject_subject_similarity=True)



