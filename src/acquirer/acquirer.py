import sys
sys.path.append("..\src")

from abc import ABC
from molecule_pool.molecule_pool import MoleculePool
import numpy as np


class Acquirer(ABC):

    def __init__(self, name, batch_size):
        self.name = name
        self.batch_size = batch_size
        self.require_var = False

    def select_train_set(self, moleculepool):
        pass


class RandomSearch(Acquirer):

    def __init__(self, batch_size):
        super().__init__("RandomSearch", batch_size)
        self.require_var = False

    def select_train_set(self, moleculepool: MoleculePool) -> MoleculePool:
        """
        Return a random subset of the molecule dataset of size batch size
        :param moleculepool:
        :return:
        """
        train_idx = np.random.choice(len(moleculepool.df), size=self.batch_size, replace=False)
        train_set = MoleculePool(moleculepool.df[train_idx])
        return train_set


class Greedy(Acquirer):
    def __init__(self, batch_size):
        super().__init__("Greedy", batch_size)
        self.require_var = False

    def select_train_set(self, moleculepool: MoleculePool) -> MoleculePool:
        """
        Select the top molecules that have the highest predicted score.
        :param moleculepool:
        :param batch_size:
        :return:
        """
        idx_best_preds = moleculepool.sort_idx_best_preds()[:self.batch_size]
        return MoleculePool(moleculepool.df[idx_best_preds])


class UBC(Acquirer):
    def __init__(self, batch_size, beta=2):
        super().__init__("UBC", batch_size)
        self.beta = beta
        self.dict_ = {}
        self.require_var = True

    def select_train_set(self, moleculepool: MoleculePool) -> MoleculePool:
        """
        Select the top molecules that have the highest UBC score, which is a combination of exploration and exploitation
        :param moleculepool:
        :param batch_size:
        :return:
        """
        ucb_score = moleculepool.score + self.beta*np.sqrt(moleculepool.variance)
        index_sorted = np.argsort(ucb_score)[:self.batch_size]
        return MoleculePool(moleculepool.df[index_sorted])