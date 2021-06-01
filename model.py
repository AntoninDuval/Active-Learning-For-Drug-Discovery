from sklearn.ensemble import RandomForestRegressor
from abc import ABC
from molecule_pool import *


class Model(ABC):
    def __init__(self, name):
        self.name = name

    def predict(self, moleculepool : MoleculePool):
        pass

    def train(self, moleculepool : MoleculePool):
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

    def predict(self, test_set: MoleculePool):
        """

        :param moleculepool:
        :return:
        """
        test_prepro = test_set.preprocess_data()
        print('R2 score: ', self.model.score(test_prepro, test_set.target))
        score = self.model.predict(test_prepro)
        test_set.add_score(score)
        return score

