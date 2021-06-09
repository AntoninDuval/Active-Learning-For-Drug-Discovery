# aqemia_test

## Overview
To develop a new drug, we often need to screen larger and larger libraries to find the molecules with the highest affinity for a given protein. To deal with this issue, Bayesian optimization techniques can be used as an active learning strategy to explore the library in an optimized way.
The goal is to find in the fewest number of iterations the maximum of top-k scored molecules, thus reducing the computation power needed for screening large library.

This repository is a simple implementation of the method proposed in the paper [Accelerating high-throughput virtual screening through molecular pool-based active learning](https://arxiv.org/abs/2012.07127). This code was used with a given dataset of 50k molecules, Enamine 50k, where the docking score is already computed for every row.

## Requirements
- Python (>= 3.6)
- PyTorch (>= 1.8.1)

## Object Model

The structure of this code follow broadly what is describe in the paper.

**MoleculePool**: Class for managing a dataset of molecules. 

**Acquirer**: An acquirer is at the core of the active learning strategy. It uses a metric for selecting the next batch to add to the training.

**Model**: The model class contains different machine and deep learning models that are trained to predict to docking score.

**Network**: Contains the architecture of deep learning models in PyTorch.

