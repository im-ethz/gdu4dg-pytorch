'''
nohup /local/home/sfoell/anaconda3/envs/gdu_torch/bin/python3.9 -u /local/home/sfoell/GitHub/gdu-pytorch/experiments/wilds/iwildcam/main.py > /local/home/sfoell/GitHub/gdu-pytorch/experiments/wilds/iwildcam/main.log 2>&1 &
'''

import os
import sys
import pickle
import torch
import numpy as np
import torchmetrics

sys.path.append("/local/home/sfoell/GitHub/gdu-pytorch")

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
from sklearn.metrics import f1_score

from gdu_pytorch.model import LayerModel
from gdu_pytorch.loss import LayerLoss
import torchvision.transforms as transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision
#from torchvision import models

import torch
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    """An identity layer"""
    def __init__(self, d):
        super().__init__()
        self.in_features = d
        self.out_features = d

    def forward(self, x):
        return x

def initialize_transform():
    transform_steps = [transforms.Resize((448, 448))]
    _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
    _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]
    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    )

    transform_steps.append(transforms.ToTensor())
    transform_steps.append(default_normalization)
    transform = transforms.Compose(transform_steps)

    return transform


def get_wilds_data(batch_size):
    # Specify the wilds dataset
    dataset = get_dataset(dataset='iwildcam', download=True)

    train_data = dataset.get_subset('train', transform=initialize_transform())
    valid_data = dataset.get_subset('val', transform=initialize_transform())
    test_data = dataset.get_subset('test', transform=initialize_transform())

    train_loader = get_train_loader('standard', train_data, batch_size=batch_size)
    valid_loader = get_train_loader('standard', valid_data, batch_size=batch_size, ) #n_groups_per_batch=2)
    test_loader = get_train_loader('standard', test_data, batch_size=batch_size,)

    return dataset, train_loader, valid_loader, test_loader


def iwildcam_classification(similarity = 'CS',
                          single_best = False,
                          single_source_domain = None,
                          batch_size = 16,
                          batch_norm = False,
                          lr = 3e-5,
                          activation = 'tanh',
                          dropout = 0.5,
                          epochs = 12,
                          patience = 10,
                          early_stopping = False,
                          bias = False,
                          fine_tune = True,
                          lambda_sparse = 0.001,
                          lambda_OLS = 0.001,
                          lambda_orth = 0.001,
                          kernel = 'RBF',
                          num_domains = 5,
                          domain_dim = 10,
                          sigma = 30,
                          softness_param = 2,
                          save_file = True,
                          save_plot = False,
                          save_feature = False,
                          run = 0,
                          results_path = None,
                          **kwargs):

    # -------------------------- read data
    dataset, train_loader, valid_loader, test_loader = get_wilds_data(batch_size=batch_size)
    # -------------------------- model
    device = init_gpu('0')
    ####################################
    model_resnet = torchvision.models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*(list(model_resnet.children())[:-1]))
    model = nn.Sequential(
        model,
        nn.Linear(2048, 182)
    )
    '''
    constructor = getattr(torchvision.models, 'resnet50')
    
    last_layer_name = 'fc'
    feature_extractor = constructor()
    d_features = getattr(feature_extractor, last_layer_name).in_features
    last_layer = Identity(d_features)
    feature_extractor.d_out = d_features
    classifier = nn.Linear(d_features, 182)

    setattr(feature_extractor, last_layer_name, classifier)
    model = feature_extractor

    # print(newmodel)
    
    model = LayerModel(
        device=device,
        task='classification',
        feature_extractor=feature_extractor,
        feature_vector_size=2048,
        output_size=182,
        num_gdus=num_domains,
        domain_dim=domain_dim,
        kernel_name=kernel,
        sigma=sigma,
        similarity_measure_name=similarity,                           # MMD, CS, Projected
        softness_param=softness_param
    )
    '''

    # TODO: check if last layer is softmax and whether loss function should be with or without logits
    ####################################
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    base_criterion = nn.CrossEntropyLoss(ignore_index=- 100, reduction = 'mean')
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
    metrics = {'accuracy':      torchmetrics.Accuracy().to(device),
               'F1-Score':      torchmetrics.F1Score(num_classes=182, avergae="macro", multiclass=True).to(device),
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
        #i = 0
        for inputs, target, metadata in tqdm(train_loader, position=0, leave=True):
            #if i <5:
            inputs, target = inputs.to(device), target.to(device)

            optimiser.zero_grad()

            output = model(inputs)
            loss = base_criterion(output, target)
            #loss = layer_criterion(output, target, model)
            batch_logs['source_train'].append(loss.item())

            loss.backward()
            optimiser.step()
                #i+=1

            #else:
               #break
        optimiser.zero_grad()

        # validation
        model.eval()
        i = 0
        with torch.no_grad():
            for inputs, target, metadata in tqdm(test_loader, position=0, leave=True):
                #if i < 2670:
                #    i+=1
                #    pass
                #else:
                inputs, target = inputs.to(device), target.to(device)
                #print(inputs.size())
                output = model(inputs)
                #loss = layer_criterion(output, target, model)
                loss = base_criterion(output, target)

                batch_logs['source_test'].append(loss.item())
                soft = nn.Softmax(dim = 1).to(device)

                batch_logs['target_test'].append(loss.item())
                batch_logs_cgm['test']['loss'].append(loss.item())
                batch_logs_cgm['test']['accuracy'].append(metrics['accuracy'](soft(output), target).item())
                #batch_logs_cgm['test']['F1-Score'].append(metrics['F1-Score'](soft(output), target).item())
                batch_logs_cgm['test']['F1-Score'].append(f1_score(target.cpu().numpy(), soft(output).cpu().numpy().argmax(axis=1), average="macro"))
                batch_logs_cgm['test']['cross_entropy'].append(metrics['cross_entropy'](output, target).item())

            print(np.mean(batch_logs_cgm['test']['F1-Score']))
            print(np.mean(batch_logs_cgm['test']['accuracy']))

        # early stopping
        for split in logs.keys():
            logs[split].append(np.mean(batch_logs[split]))

    all_y_pred, all_y_true, all_metadata = [], [], []
    with torch.no_grad():
        for inputs, target, metadata in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            output = model(inputs)
            all_metadata.append(metadata)
            all_y_pred.append(output)
            all_y_true.append(target)

    all_y_pred, all_y_true, all_metadata = torch.argmax(torch.cat(all_y_pred, dim = 0)), torch.cat(all_y_true, dim = 0), torch.cat(all_metadata, dim = 0)
    eval_df = dataset.eval(all_y_pred.cpu(), all_y_true.cpu(), all_metadata.cpu())
    print(eval_df)

    return eval_df



if __name__ == '__main__':
    print(args)
    iwildcam_classification()
    # define hyperparameters
    #hyperparams = {k: v for k, v in vars(args).items() if type(v) == list}
    #print("Hyperparams: ", hyperparams)
    #grid = ParameterGrid(hyperparams)

    #for params in grid:
    #    experiment = {**vars(args), **params}
    #    print(experiment)
    #    try:
    #        digits_classification(**experiment)
    #    except:
    #        traceback.print_exc()
    #        pass
