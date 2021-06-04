from abc import ABC
import pandas as pd
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

        df = moleculepool.df.sample(n=self.batch_size)
        train_set = MoleculePool(df)
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
        index_sorted = np.argsort(moleculepool.score)
        return MoleculePool(moleculepool.df.iloc[index_sorted[:self.batch_size]])


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