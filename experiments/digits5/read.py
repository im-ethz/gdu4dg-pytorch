import os
import torch
import numpy as np

from scipy.io import loadmat
import torch.nn.functional as F
from torchvision.transforms import Resize

def load_mnist(data_dir):
    data = loadmat(os.path.join(data_dir, "mnist_data.mat"))
    train_image = np.reshape(data['train_32'], (55000, 32, 32, 1))
    test_image = np.reshape(data['test_32'], (10000, 32, 32, 1))

    # turn to the 3 channel image with C*H*W
    train_image = np.concatenate([train_image, train_image, train_image], 3)
    test_image = np.concatenate([test_image, test_image, test_image], 3)
    train_image = train_image.transpose(0, 3, 1, 2).astype(np.float32)
    test_image = test_image.transpose(0, 3, 1, 2).astype(np.float32)

    # get labels
    train_label = np.argmax(data['label_train'], axis=1)
    test_label = np.argmax(data['label_test'], axis=1)

    return {'train': {'image': train_image, 'label': train_label}, 'test': {'image': test_image, 'label': test_label}}

def load_mnist_m(data_dir):
    data = loadmat(os.path.join(data_dir, "mnistm_with_label.mat"))
    train_image = data['train']
    test_image = data['test']
    train_image = train_image.transpose(0, 3, 1, 2).astype(np.float32)
    test_image = test_image.transpose(0, 3, 1, 2).astype(np.float32)

    # get labels
    train_label = np.argmax(data['label_train'], axis=1)
    test_label = np.argmax(data['label_test'], axis=1)

    return {'train': {'image': train_image, 'label': train_label}, 'test': {'image': test_image, 'label': test_label}}

def load_svhn(data_dir):
    train_data = loadmat(os.path.join(data_dir, "svhn_train_32x32.mat"))
    test_data = loadmat(os.path.join(data_dir, "svhn_test_32x32.mat"))
    train_image = train_data['X']
    test_image = test_data['X']
    train_image = train_image.transpose(3, 2, 0, 1).astype(np.float32)
    test_image = test_image.transpose(3, 2, 0, 1).astype(np.float32)

    # get labels
    train_label = train_data["y"].reshape(-1)
    test_label = test_data["y"].reshape(-1)

    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0

    return {'train': {'image': train_image, 'label': train_label}, 'test': {'image': test_image, 'label': test_label}}

def load_syn(data_dir):
    # note: in tensorflow implementation, this dataset did not undergo permutation
    train_data = loadmat(os.path.join(data_dir, "synth_train_32x32.mat"))
    test_data = loadmat(os.path.join(data_dir, "synth_test_32x32.mat"))
    train_image = train_data["X"]
    test_image = test_data["X"]
    train_image = train_image.transpose(3, 2, 0, 1).astype(np.float32)
    test_image = test_image.transpose(3, 2, 0, 1).astype(np.float32)

    # get labels
    train_label = train_data["y"].reshape(-1)
    test_label = test_data["y"].reshape(-1)

    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0

    return {'train': {'image': train_image, 'label': train_label}, 'test': {'image': test_image, 'label': test_label}}

def load_usps(data_dir):
    data = loadmat(os.path.join(data_dir, "usps_28x28.mat"))["dataset"]
    train_image = data[0][0] * 255 # TODO: not necessary because we normalize?
    test_image = data[1][0] * 255

    train_image = np.concatenate([train_image, train_image, train_image], 1)
    test_image = np.concatenate([test_image, test_image, test_image], 1)

    # get labels
    train_label = data[0][1].reshape(-1)
    test_label = data[1][1].reshape(-1)

    train_label[train_label == 10] = 0
    test_label[test_label == 10] = 0

    # repeat the training set # TODO: why do we want to repeat the dataset?
    train_image = np.tile(train_image, (4, 1, 1, 1)) 
    train_label = np.tile(train_label, 4)

    return {'train': {'image': train_image, 'label': train_label}, 'test': {'image': test_image, 'label': test_label}}

def resample(data, train_size=25000, test_size=9000):
    # random sample 25000 from train dataset and random sample 9000 from test dataset
    inds = np.random.permutation(data['train']['image'].shape[0])
    data['train']['image'] = data['train']['image'][inds]
    data['train']['label'] = data['train']['label'][inds]

    data['train']['image'] = data['train']['image'][:train_size]
    data['train']['label'] = data['train']['label'][:train_size]
    data['test']['image'] = data['test']['image'][:test_size]
    data['test']['label'] = data['test']['label'][:test_size]

    return data

def transform(data, img_shape=32):
    # resize image
    data['train']['image'] = Resize(img_shape)(torch.tensor(np.uint8(data['train']['image'])))
    data['test']['image'] = Resize(img_shape)(torch.tensor(np.uint8(data['test']['image'])))

    # onehot encode labels
    data['train']['label'] = F.one_hot(torch.tensor(data['train']['label'], dtype=torch.int64), num_classes=10)
    data['test']['label'] = F.one_hot(torch.tensor(data['test']['label'], dtype=torch.int64), num_classes=10)

    return data

def load_digit5(domain, data_dir, train_size, test_size):
    if domain == "mnist":
        data = load_mnist(data_dir)
    elif domain == "mnistm":
        data = load_mnist_m(data_dir)
    elif domain == "svhn":
        data = load_svhn(data_dir)
    elif domain == "syn":
        data = load_syn(data_dir)
    elif domain == "usps":
        data = load_usps(data_dir)
    else:
        raise NotImplementedError("Domain {} Not Implemented".format(domain))

    data = resample(data, train_size, test_size)

    data = transform(data)

    return data