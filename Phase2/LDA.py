from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from sklearn.preprocessing import scale

'''
Base class for LDA decomposition.
'''


class LDAModel:
    def __init__(self, database_matrix, k_components, image_list):
        self.database_matrix = database_matrix
        self.k_components = k_components
        self.lda = LatentDirichletAllocation(n_components=k_components, random_state=42, max_iter=15)
        self.database_image_list = image_list
        self.reduced_database_matrix = None

    '''
        This is the default method which does the decomposition.
    '''

    def decompose(self):
        self.reduced_database_matrix = self.lda.fit_transform(self.database_matrix)
        self.get_decomposed_data_matrix()

    def get_feature_latent_semantic_term_weight_sorted(self, idx):
        latent_semantic = self.lda.components_[idx]
        term_weight_dict = {}
        for index, value in enumerate(latent_semantic, 1):
            term_weight_dict["feature" + str(index)] = value

        term_weight_dict_sorted = sorted(term_weight_dict.items(), key=lambda kv: kv[1], reverse=True)
        return term_weight_dict_sorted

    def get_data_latent_semantic_term_weight_sorted(self, idx):
        latent_semantic = self.reduced_database_matrix[:, idx]
        term_weight_dict = {}
        for image_id, value in zip(self.database_image_list, latent_semantic):
            term_weight_dict[image_id] = value

        term_weight_dict_sorted = sorted(term_weight_dict.items(), key=lambda kv: kv[1], reverse=True)
        return term_weight_dict_sorted

    def get_decomposed_data_matrix(self):
        self.reduced_database_matrix = self.reduced_database_matrix[:, :self.k_components]
        return self.reduced_database_matrix

    def print_term_weight_pairs(self, k=-1):
        for idx in range(k):
            print("Printing data-latentsemantics term-weight pairs for Data-Latentsemantic - " + str(idx))
            data_latentsemantics_term_weight = self.get_data_latent_semantic_term_weight_sorted(idx)
            print(data_latentsemantics_term_weight)
            print("---------------------------------------------------------------------------------------")
            print("Printing feature-latentsemantics term-weight pairs for Feature-Latentsemantic - " + str(idx))
            feature_latentsemantics_term_weight = self.get_feature_latent_semantic_term_weight_sorted(idx)
            print(feature_latentsemantics_term_weight)
            print("****************************************************************************************")

    def get_new_image_features_in_latent_space(self, image_features):
        latent_features = self.lda.transform(image_features)
        return latent_features[0][:self.k_components]




