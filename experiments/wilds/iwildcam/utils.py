import torch
import torch.nn as nn
from torchmetrics.functional import accuracy


dassl_feature_extractor = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same'),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
)


domain_net_feature_extractor = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same'),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(1, 1), padding='same'),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

    nn.Flatten(),
    nn.Linear(2048, 3072),
    nn.BatchNorm1d(3072),
    nn.ReLU(),
    nn.Dropout(p=0.5),

    nn.Linear(3072, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU()
)


def calc_accuracy(pred_one_hot, true_one_hot):
    pred_labels = torch.max(pred_one_hot, dim=1)[1]
    true_labels = torch.max(true_one_hot, dim=1)[1]
    acc = 100 * accuracy(pred_labels, true_labels).item()
    return acc
