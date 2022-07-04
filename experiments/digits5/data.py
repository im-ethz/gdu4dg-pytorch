import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize
import numpy as np

class DigitFiveDataset(Dataset):
    def __init__(self, image, label):
        self.image, self.label = image, label
        self.transform = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __getitem__(self, idx):
        img, label = self.image[idx], self.label[idx]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return self.image.shape[0]