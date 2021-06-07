import pandas as pd
from acquirer.acquirer import *
from model.model import *
from molecule_pool.molecule_pool import MoleculePool

BATCH_SIZE = 500
MAX_ITERATIONS = 6
k = 500

df = pd.read_csv('../data/Enamine50k_rdkit2d.csv').to_numpy()
df = df[:, 1:]

molecule_pool = MoleculePool(df)

model = RandomForest({'n_estimators': 100,
                      'max_depth': 8,
                      })

model = NN({'input_shape': 200, 'hidden_shape': 100, 'output_shape': 1}, epoch=100)

acquisition_function = Greedy(batch_size=BATCH_SIZE)


print('Initialize training set...')
train_set, test_set = molecule_pool.initialize_batch(batch_size=BATCH_SIZE)

iteration = 0

# Get the name of top k molecule according to the docking score score
idx_best = molecule_pool.sort_idx_by_true_score()[:k]
top_k_mol = set(molecule_pool.df[idx_best, 0])

top_k_found = train_set.get_top_k(k, top_k_mol)
print("% of top {} molecules found :".format(k), (len(top_k_found)/k)*100, '%')

while iteration < MAX_ITERATIONS:
    print('ITERATION', iteration)
    print('Train set shape : ', train_set.data.shape)

    print('Training the model...')
    model.train(train_set)

    print('Evaluating the model...')
    score = model.predict(test_set, acquisition_function.require_var)
    most_promising_mol = acquisition_function.select_train_set(test_set)

    new_train_mol = np.concatenate((train_set.df[:, 0], most_promising_mol.df[:, 0]))

    train_set, test_set = molecule_pool.create_batch(new_train_mol)

    iteration += 1

    print('Get top k molecules...')

    top_k_found = train_set.get_top_k(k, top_k_mol)

    print("% of top {} molecules found :".format(k), (len(top_k_found)/k)*100, '%')

    print('='*50)

