import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
from tqdm import tqdm
from PIL import Image

class DigitFiveDataset(Dataset):
    def __init__(self, data, split):
        self.image, self.label = data[split]['image'], data[split]['label']

        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.label = F.one_hot(torch.tensor(self.label, dtype=torch.int64), num_classes=10)

    def __getitem__(self, idx):
        img, label = self.image[idx], self.label[idx]

        img = Image.fromarray(np.uint8(np.asarray(img.transpose((1, 2, 0)))))
        img = self.transform(img)

        return img, label

    def __len__(self):
        return self.image.shape[0]