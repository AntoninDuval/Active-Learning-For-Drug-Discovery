from abc import ABC
from molecule_pool import MoleculePool
import numpy as np


class Acquirer(ABC):

    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size

    def select_train_set(self, moleculepool):
        pass


class RandomSearch(Acquirer):

    def __init__(self, batch_size):
        super().__init__("RandomSearch", batch_size)

    def select_train_set(self, moleculepool: MoleculePool):

        train_idx = np.random.choice(len(moleculepool.df), size=self.batch_size, replace=False)
        train_set = MoleculePool(moleculepool.df[train_idx])
        return train_set


class Greedy(Acquirer):
    def __init__(self, batch_size):
        super().__init__("Greedy", batch_size)

    def select_train_set(self, moleculepool: MoleculePool):
        """

        :param moleculepool:
        :param batch_size:
        :return:
        """
        idx_best_preds = moleculepool.sort_idx_best_preds()[:self.batch_size]
        return MoleculePool(moleculepool.df[idx_best_preds])


class UBC(Acquirer):
    def __init__(self, batch_size, beta=2):
        super().__init__("Uncertainty", batch_size)
        self.beta = beta
        self.dict_ = {}

    def select_train_set(self, moleculepool: MoleculePool):
        """

        :param moleculepool:
        :param batch_size:
        :return:
        """
        ucb_score = moleculepool.score + self.beta*np.sqrt(moleculepool.variance)
        index_sorted = np.argsort(ucb_score)
        return MoleculePool(moleculepool.df.iloc[index_sorted[:self.batch_size]])