from sklearn.ensemble import RandomForestRegressor
from abc import ABC
from molecule_pool.molecule_pool import *
import numpy as np
import torch
import torch.nn as nn
from network.mlp import MLP
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm


class Model(ABC):
    def __init__(self, name):
        self.name = name

    def predict(self, moleculepool: MoleculePool, require_var: bool):
        pass

    def train(self, moleculepool : MoleculePool, show_error=True):
        pass


class RandomForest(Model):
    def __init__(self, param):
        super().__init__("RandomForestRegressor")
        self.model = RandomForestRegressor(**param)

    def train(self, moleculepool):
        """

        :param moleculepool:
        :return:
        """
        data = moleculepool.preprocess_data()
        target = moleculepool.target
        self.model.fit(data, target)

        print('R2 score on train: ', self.model.score(data, target))

    def predict(self, test_set: MoleculePool, require_var, verbose:bool=False):
        """

        :param moleculepool:
        :return:
        """
        preds = np.zeros((len(test_set.df), len(self.model.estimators_)))

        test_prepro = test_set.preprocess_data()
        if verbose:
            print('R2 score: ', self.model.score(test_prepro, test_set.target))

        score = self.model.predict(test_prepro)

        if require_var:
            print('Computing the variance...')
            for j, submodel in tqdm.tqdm(enumerate(self.model.estimators_)):
                preds[:, j] = submodel.predict(test_prepro)
            test_set.add_variance(np.var(preds, axis=1))

        test_set.add_score(score)
        return score


class NN(Model):

    def __init__(self, param, epoch):
        super().__init__("NN")
        self.model = MLP(**param).double()
        self.optimiser = Adam(self.model.parameters(), lr=0.01, weight_decay=0.01)
        self.epoch = epoch
        self.criterion = nn.MSELoss()

    def train(self, moleculepool, verbose:bool=False):
        """
        :param moleculepool:
        :return:
        """
        data = moleculepool.preprocess_data().astype(float)
        inputs = torch.tensor(np.concatenate([data, moleculepool.target.reshape(-1, 1).astype(float)], axis=1))

        dataloader = DataLoader(
            inputs,
            batch_size=4096,
            shuffle=False,
        )

        for _ in range(self.epoch):
            for i, batch in enumerate(dataloader):
                self.optimiser.zero_grad()
                x = batch[:, :-1]
                y = batch[:, -1]
                y = self._normalize(y)
                preds = self.model(x.double())
                loss = self.criterion(preds.squeeze(), y)
                loss.backward()
                self.optimiser.step()
        if verbose:
            print(loss)

    def predict(self, test_set: MoleculePool, require_var=False):

        test_prepro = torch.tensor(test_set.preprocess_data().astype(float))
        with torch.no_grad():
            score = self.model(test_prepro.detach())

        score = score * self.std + self.mean
        test_set.add_score(score)
        return score

    def reset_params(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def _normalize(self, target):
        self.mean = np.nanmean(target)
        self.std = np.nanstd(target)
        return (target - self.mean) / self.std


