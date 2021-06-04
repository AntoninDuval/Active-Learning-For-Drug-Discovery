import pandas as pd
from acquirer import *
from model import *
from molecule_pool import MoleculePool

BATCH_SIZE = 500
MAX_ITERATIONS = 6
k = 500

df = pd.read_csv('Enamine50k_rdkit2d.csv')
df = df.iloc[:, 1:]

molecule_pool = MoleculePool(df)

model = RandomForest({'n_estimators': 100,
                      'max_depth': 8,
                      })

acquisition_function = UBC(batch_size=BATCH_SIZE)


print('Initialize training set...')
train_set, test_set = molecule_pool.initialize_batch(batch_size=BATCH_SIZE)

iteration = 0

# Get the top k molecule according to the docking score score
real_top_k = molecule_pool.df.sort_values("score")[:k]

while iteration < MAX_ITERATIONS:
    print('ITERATION', iteration)
    print('Train set shape : ', train_set.df.shape)

    print('Training the model...')
    model.train(train_set)

    print('Evaluating the model...')
    score = model.predict(test_set)

    most_promising_mol = acquisition_function.select_train_set(test_set)

    new_train_index = train_set.df.index.append(most_promising_mol.df.index)

    train_set, test_set = molecule_pool.create_batch(new_train_index)

    iteration += 1

    print('Get top k molecules...')
    #model.predict(molecule_pool, show_r2=False)
    top_k = train_set.new_get_top_k(k, real_top_k)

    print('='*50)

