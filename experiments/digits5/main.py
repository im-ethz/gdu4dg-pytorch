import torch
import numpy as np

from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from torch import nn
import traceback
import copy
from tqdm import tqdm

from config import args, init_gpu
from data import DigitFiveDataset
from read import load_digit5

def digits_classification(data_dir,
                          TARGET_DOMAIN,
                          SOURCE_SAMPLE_SIZE,
                          TARGET_SAMPLE_SIZE,
                          similarity,
                          single_best = False,
                          single_source_domain = None,
                          batch_size = 128,
                          batch_norm = False,
                          lr = 0.001,
                          activation = 'tanh',
                          dropout = 0.5,
                          epochs = 100,
                          patience = 10,
                          early_stopping = True,
                          bias = False,
                          fine_tune = True,
                          lambda_sparse = 0,
                          lambda_OLS = 0,
                          lambda_orth = 0,
                          kernel = None,
                          num_domains = 5,
                          domain_dim = 10,
                          sigma = 7.5,
                          softness_param = 2,
                          save_file = True,
                          save_plot = False,
                          save_feature = False,
                          **kwargs):

    # -------------------------- read data
    SOURCE_DOMAINS = ('mnist', 'mnistm', 'svhn', 'syn', 'usps')

    data = {s:load_digit5(s, data_dir, SOURCE_SAMPLE_SIZE, TARGET_SAMPLE_SIZE) for s in SOURCE_DOMAINS}

    x_source_train = torch.cat([data[s]['train']['image'] for s in data.keys() if s.lower() != TARGET_DOMAIN.lower()], axis=0)
    y_source_train = torch.cat([data[s]['train']['label'] for s in data.keys() if s.lower() != TARGET_DOMAIN.lower()], axis=0)

    x_source_test = torch.cat([data[s]['test']['image'] for s in data.keys() if s.lower() != TARGET_DOMAIN.lower()], axis=0)
    y_source_test = torch.cat([data[s]['test']['label'] for s in data.keys() if s.lower() != TARGET_DOMAIN.lower()], axis=0)

    x_target_test = data[TARGET_DOMAIN]['test']['image']
    y_target_test = data[TARGET_DOMAIN]['test']['label']

    data_source_train = DigitFiveDataset(x_source_train, y_source_train)
    data_source_test = DigitFiveDataset(x_source_test, y_source_test)
    data_target_test = DigitFiveDataset(x_target_test, y_target_test)

    source_train_loader = DataLoader(data_source_train, batch_size=batch_size, shuffle=True)
    source_test_loader = DataLoader(data_source_test, batch_size=batch_size, shuffle=True)
    target_test_loader = DataLoader(data_target_test, batch_size=batch_size, shuffle=True)

    # -------------------------- model
    device = init_gpu(0)
    ####################################
    # TODO Andras: insert model layer here

    # model = ....
    # TODO: check if last layer is softmax and whether loss function should be with or without logits
    ####################################

    model = nn.DataParallel(model).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # -------------------------- train
    min_loss = np.inf
    best_model = None
    logs = {split: [] for split in ('source_train', 'source_test', 'target_test')}

    # TODO: in tensorflow code it seems like they only train for one epoch in the beginning
    for e in tqdm(range(epochs)):

        batch_logs = {split: [] for split in ('source_train', 'source_test', 'target_test')}

        # training
        model.train()
        for inputs, target in source_train_loader:
            inputs, target = inputs.to(device), target.to(device)

            optimiser.zero_grad()

            output = model(inputs)
            loss = criterion(output, target)

            batch_logs['source_train'].append(loss.item())

            loss.backward()
            optimiser.step()

        optimiser.zero_grad()

        # validation
        model.eval()
        with torch.no_grad():
            for inputs, target in source_test_loader:
                inputs, target = inputs.to(device), target.to(device)

                output = model(inputs)
                loss = criterion(output, target)

                batch_logs['source_test'].append(loss.item())

            for inputs, target in target_test_loader:
                inputs, target = inputs.to(device), target.to(device)

                output = model(inputs)
                loss = criterion(output, target)

                batch_logs['target_test'].append(loss.item())

        # early stopping
        for split in logs.keys():
            logs[split].append(np.mean(batch_logs[split]))

        if logs['source_test'][-1] < min_loss:
            min_loss = logs['source_test'][-1]
            best_model = copy.deepcopy(model)

        elif np.array(logs['source_test'][-patience:] > min_loss).sum() == patience:
            print("Early stopping!")
            break

    # -------------------------- fine-tune
    # TODO if fine_tune and similarity == None:


if __name__ == '__main__':
    print(args)

    # define hyperparameters
    hyperparams = {k:v for k, v in vars(args).items() if type(v) == list}
    print("Hyperparams: ", hyperparams)
    grid = ParameterGrid(hyperparams)

    for params in grid:
        experiment = {**vars(args), **params}
        print(experiment)
        try:
            digits_classification(**experiment)
        except:
            traceback.print_exc()
            pass