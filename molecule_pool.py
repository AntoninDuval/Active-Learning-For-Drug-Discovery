import pandas as pd
import numpy as np

class MoleculePool:

    def __init__(self, df):
        self.df = df
        self.target = self.df['score']
        self.data = self.df.drop('score', axis=1)
        self.score = None

    def initialize_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        train = MoleculePool(self.df.sample(n=batch_size))
        test = MoleculePool(self.df.drop(train.df.index, axis=0))
        return train, test

    def create_batch(self, train_index):
        """
        Create a new training and testing set from a list of indexes
        :param train_index:
        :return:
        """
        train = MoleculePool(self.df.iloc[train_index])
        test = MoleculePool(self.df.drop(train.df.index, axis=0))

        return train, test

    def get_top_k(self, k):
        real_score_sorted = np.argsort(self.target)
        predicted_score_sorted = np.argsort(self.score)

        real_top_k = MoleculePool(self.df.iloc[real_score_sorted[:k]])
        predicted_top_k = MoleculePool(self.df.iloc[predicted_score_sorted[:k]])

        top_k_found = len(predicted_top_k.df[predicted_top_k.df["name"].isin(real_top_k.df["name"])])
        print('Top', k, 'molecules found : ', top_k_found)

    def new_get_top_k(self, k, real_top_k):
        # Get best molecule
        found_top_k = self.df.sort_values('score')[:k]
        matched_top_k = real_top_k.merge(found_top_k, how='inner')
        print(len(matched_top_k))

    def preprocess_data(self):
        preprocessed_data = self.data.drop(["name", "smiles"], axis=1)
        return preprocessed_data

    def add_score(self, score):
        self.score = score