import os
import sys
import time
import json

import numpy as np

from main import iwildcam_classification

sys.path.append(os.path.expanduser("~/Thesis/layer/"))

args = {'data_dir': '/wave/odin/digitfive/',
        'similarity': 'CS',
        'batch_size': 16,
        'batch_norm': False,
        'lr': 0.001,
        'activation': 'tanh',
        'dropout': 0.5,
        'epochs': 1,
        'patience': 10,
        'early_stopping': True,
        'bias': False,
        'fine_tune': False,
        'lambda_sparse': 0.001,
        'lambda_OLS': 0.001,
        'lambda_orth': 0,
        'kernel': None,
        'num_domains': 5,
        'domain_dim': 10,
        'sigma': 7.5,
        'softness_param': 2,
        'save_file': True,
        'save_plot': 2,
        'save_feature': False,
        'results_path': None
        }

hp_grid = {
    'CS': {'similarity': 'CS', 'lambda_orth': 0},
    'MMD': {'similarity': 'MMD', 'lambda_orth': 0},
    'Projected': {'similarity': 'Projected', 'lambda_orth': 1e-8}
}

folder_name = f"experiments_{time.strftime('%Y%m%d-%H%M%S')}"
experiments_result_path = os.path.join('results', folder_name)
if not os.path.exists(experiments_result_path):
    os.makedirs(experiments_result_path)

with open(os.path.join(experiments_result_path, 'hp_grid.txt'), 'w') as file:
    file.write(json.dumps(hp_grid))


for key, value in hp_grid.items():
    print(f"HPs used: {value}")
    args['similarity'] = value['similarity']
    args['lambda_orth'] = value['lambda_orth']
    experiment_path = os.path.join(experiments_result_path, args['similarity'])
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    args['results_path'] = experiment_path

    results = {'loss': [], 'accuracy': [], 'cross_entropy': []}
    for run in range(3):
        print(f"run: {run}")
        args['run'] = run
        eval_df = iwildcam_classification(**args)
        results['loss'].append(loss)
        results['accuracy'].append(accuracy)
        results['cross_entropy'].append(cross_entropy)
        results['F1-Score'].append(cross_entropy)
        results['mean_accuracy'] = np.mean(results['accuracy'])
        results['std_accuracy'] = np.std(results['accuracy'])
        with open(os.path.join(experiment_path, f'results_{run}.txt'), 'w') as file:
            file.write(json.dumps(results))

        print(f"Run: {run} finished \n"
              f"Accuracies: {results['accuracy']} \n"
              f"Mean Accuracy: {results['mean_accuracy']} +- ({results['std_accuracy']}) \n")
