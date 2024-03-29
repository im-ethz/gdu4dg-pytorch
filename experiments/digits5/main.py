import os
import pickle
import torch
import numpy as np

from sklearn.model_selection import ParameterGrid
from torch.utils.data import DataLoader
from torch import nn
from torchmetrics import ConfusionMatrix, Accuracy, Precision, Recall, Specificity, PrecisionRecallCurve, AUC, AUROC
import traceback
import copy
from tqdm import tqdm

from config import args, init_gpu
from data import DigitFiveDataset
from read import load_digit5
from utils import domain_net_feature_extractor, dassl_feature_extractor, calc_accuracy
from gdu_pytorch.model import LayerModel
from gdu_pytorch.loss import LayerLoss


def digits_classification(data_dir,
                          TARGET_DOMAIN,
                          SOURCE_SAMPLE_SIZE,
                          TARGET_SAMPLE_SIZE,
                          similarity = 'CS',
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
                          kernel = 'RBF',
                          num_domains = 5,
                          domain_dim = 10,
                          sigma = 7.5,
                          softness_param = 2,
                          save_file = True,
                          save_plot = False,
                          save_feature = False,
                          run = 0,
                          results_path = None,
                          **kwargs):

    # -------------------------- read data
    SOURCE_DOMAINS = ('mnist', 'mnistm', 'svhn', 'syn', 'usps')

    data = {s: load_digit5(s, data_dir, SOURCE_SAMPLE_SIZE, TARGET_SAMPLE_SIZE) for s in SOURCE_DOMAINS}

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
    target_test_loader = DataLoader(data_target_test, batch_size=TARGET_SAMPLE_SIZE, shuffle=True)

    # -------------------------- model
    device = init_gpu('0')
    ####################################
    feature_extractor = domain_net_feature_extractor
    model = LayerModel(
        device=device,
        task='classification',
        feature_extractor=feature_extractor,
        feature_vector_size=2048,
        output_size=10,
        num_gdus=num_domains,
        domain_dim=domain_dim,
        kernel_name=kernel,
        sigma=sigma,
        similarity_measure_name=similarity,                           # MMD, CS, Projected
        softness_param=softness_param
    )

    # TODO: check if last layer is softmax and whether loss function should be with or without logits
    ####################################
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    base_criterion = nn.CrossEntropyLoss()
    layer_criterion = LayerLoss(
        device=device,
        criterion=base_criterion,
        sigma=sigma,
        lambda_OLS=lambda_OLS,
        lambda_orth=lambda_orth,
        lambda_sparse=lambda_sparse,
        orthogonal_loss=True,
        sparse_coding=True
    )
    metrics = {'accuracy':      calc_accuracy,
               'cross_entropy': nn.CrossEntropyLoss().to(device)}

    # -------------------------- train
    min_loss = np.inf
    best_models_acc = 0
    best_models_cross_entropy = 0
    best_model = None
    logs = {split: [] for split in ('source_train', 'source_test', 'target_test')}
    logs_cgm = {i: {j: [] for j in ('loss',) + tuple(metrics.keys())} for i in ('train', 'val', 'test')} # epoch logs

    # TODO: in tensorflow code it seems like they only train for one epoch in the beginning
    for e in tqdm(range(epochs)):

        batch_logs = {split: [] for split in ('source_train', 'source_test', 'target_test')}
        batch_logs_cgm = {i: {j: [] for j in logs_cgm[i].keys()} for i in logs_cgm.keys()}

        # training
        model.train()
        for inputs, target in source_train_loader:
            inputs, target = inputs.to(device), target.to(device)

            optimiser.zero_grad()

            output = model(inputs)
            loss = layer_criterion(output, target, model)

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
                loss = layer_criterion(output, target, model)

                batch_logs['source_test'].append(loss.item())

            for inputs, target in target_test_loader:
                inputs, target = inputs.to(device), target.to(device)

                output = model(inputs)
                loss = layer_criterion(output, target, model)

                batch_logs['target_test'].append(loss.item())
                batch_logs_cgm['test']['loss'].append(loss.item())
                batch_logs_cgm['test']['accuracy'].append(metrics['accuracy'](output, target))
                batch_logs_cgm['test']['cross_entropy'].append(metrics['cross_entropy'](output, target).item())

        current_test_accuracy = np.mean(batch_logs_cgm['test']['accuracy'])
        current_test_cross_entropy = np.mean(batch_logs_cgm['test']['cross_entropy'])

        # early stopping
        for split in logs.keys():
            logs[split].append(np.mean(batch_logs[split]))

        if logs['source_test'][-1] < min_loss:
            min_loss = logs['source_test'][-1]
            best_model = copy.deepcopy(model)
            best_models_acc = current_test_accuracy
            best_models_cross_entropy = current_test_cross_entropy

        elif np.array(logs['source_test'][-patience:] > min_loss).sum() == patience:
            print("Early stopping!")
            break

    # save model and logs
    models_path = os.path.join(results_path, "models")
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    torch.save(best_model.state_dict(), models_path + f"/run_{run}.pth")

    return min_loss, best_models_acc, best_models_cross_entropy


if __name__ == '__main__':
    print(args)

    # define hyperparameters
    hyperparams = {k: v for k, v in vars(args).items() if type(v) == list}
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
