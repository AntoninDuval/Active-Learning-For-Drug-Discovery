import numpy as np
from sklearn.preprocessing import StandardScaler

class MoleculePool(object):

    def __init__(self, df):
        self.df = df
        self.target = self.df[:, 2]
        self.data = np.delete(self.df, 2, axis=1)
        self.score = None
        self.variance = None

    def initialize_batch(self, batch_size: int):
        """

        :param batch_size:
        :return:
        """
        train_idx = np.random.choice(len(self.df), size=batch_size, replace=False)
        train = MoleculePool(self.df[train_idx])
        test = MoleculePool(np.delete(self.df, train_idx, axis=0))
        return train, test

    def create_batch(self, train_index):
        """
        Create a new training and testing set from a list of indexes
        :param train_index:
        :return:
        """
        train = MoleculePool(self.df[np.isin(self.df[:, 0], train_index)])
        test = MoleculePool(self.df[~np.isin(self.df[:, 0], train_index)])

        return train, test

    def get_top_k(self, k, top_k):
        # Get best molecule
        idx_best = self.sort_idx_by_true_score()[:k]
        found_top_k = self.df[idx_best, 0]

        matched_top_k = set(found_top_k).intersection(top_k)
        return matched_top_k

    def preprocess_data(self):
        preprocessed_data = np.delete(self.data, [0, 1], axis=1)
        scaler = StandardScaler()
        preprocessed_data = scaler.fit_transform(preprocessed_data)
        return preprocessed_data

    def add_score(self, score):
        self.score = score

    def add_variance(self, variance):
        self.variance = variance

    def sort_idx_by_true_score(self):
        return self.target.argsort()

    def sort_idx_best_preds(self):
        return self.score.argsort()
