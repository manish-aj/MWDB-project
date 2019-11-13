import sys
sys.path.insert(1, '../Phase1')
from features_images import FeaturesImages
import misc
import os
from pathlib import Path
from PCA import PCAModel
from SVD import SVD
from NMF import NMFModel
from LDA import LDAModel


class Decomposition:
    def __init__(self, decomposition_name='', k_components=10, feature_extraction_model_name=None, test_folder_path=None,
                 metadata_images_list=None, metadata_label=None):
        self.decomposition_name = decomposition_name
        self.k_components = k_components
        self.decomposition_model = None
        self.feature_extraction_model_name = feature_extraction_model_name
        self.test_folder_path = test_folder_path
        self.feature_extraction_object = FeaturesImages(self.feature_extraction_model_name, self.test_folder_path)
        self.feature_extraction_model = self.feature_extraction_object.get_model()
        self.database_matrix = []
        self.database_image_id = []
        self.reduced_pickle_file_folder = os.path.join(os.path.dirname(__file__), 'pickle_files')
        self.metadata_images_list = metadata_images_list
        self.metadata_label = metadata_label or ''
        self.set_database_matrix()

    def set_database_matrix(self):
        parent_directory_path = Path(os.path.dirname(__file__)).parent
        pickle_file_directory = os.path.join(parent_directory_path, 'Phase1')
        print('Getting the Model Features from Phase1')
        self.feature_extraction_object.compute_features_images_folder()
        database_images_features = misc.load_from_pickle(pickle_file_directory, self.feature_extraction_model_name)
        if self.metadata_images_list is not None:
            print("Taking images based on metadata")
            for image_id in self.metadata_images_list:
                self.database_matrix.append(database_images_features[image_id])
                self.database_image_id.append(image_id)
        else:
            for image_id, feature_vector in database_images_features.items():
                self.database_matrix.append(feature_vector)
                self.database_image_id.append(image_id)

    def dimensionality_reduction(self):
        # self.set_database_matrix()
        # Note : when we have number of images <=20 or features <=20 , we are getting an error
        # this is because the database_matrix has <=20 images and the reduction models,
        # should have n_components parameters <= n,m
        # Hence, we have to take the min(min(len(self.database_matrix[0]),len(self.database_matrix)),20)
        if self.decomposition_name == 'PCA':
            self.decomposition_model = PCAModel(self.database_matrix, self.k_components, self.database_image_id)
        elif self.decomposition_name == 'SVD':
            self.decomposition_model = SVD(self.database_matrix, self.k_components, self.database_image_id)
        elif self.decomposition_name == 'NMF':
            self.decomposition_model = NMFModel(self.database_matrix, self.k_components, self.database_image_id)
        elif self.decomposition_name == 'LDA':
            self.decomposition_model = LDAModel(self.database_matrix, self.k_components, self.database_image_id)

        self.decomposition_model.decompose()
        print('Decomposition Complete')
        decomposed_database_matrix = self.decomposition_model.get_decomposed_data_matrix()
        reduced_dimension_folder_images_dict = {}
        for image_id, reduced_feature_vector in zip(self.database_image_id, decomposed_database_matrix):
            reduced_dimension_folder_images_dict[image_id] = reduced_feature_vector
        if self.metadata_label != '':
            misc.save2pickle(reduced_dimension_folder_images_dict, self.reduced_pickle_file_folder,
                             feature=(self.feature_extraction_model_name+'_'+self.decomposition_name+
                                      '_' + self.metadata_label))
        else:
            misc.save2pickle(reduced_dimension_folder_images_dict, self.reduced_pickle_file_folder,
                             feature=(self.feature_extraction_model_name + '_' + self.decomposition_name))


