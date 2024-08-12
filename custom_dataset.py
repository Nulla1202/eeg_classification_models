import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.signal import resample

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label_index = self.data[idx]
        label = nn.functional.one_hot(torch.tensor(label_index), num_classes=2)
        image = image[:1000,:]
        image = resample(image, 500, axis=0)
        image = (image - np.mean(image, axis=0, keepdims=True)) / np.std(image, axis=0, keepdims=True)
        image = torch.from_numpy(image)
        return image.double(), label.double()

def load_all_data(directories):
    data = []
    for file in directories:
            npz_data = np.load(file)
            ecog_data = npz_data['ecog_signal_long']
            labels = npz_data['label']
            for i in range(ecog_data.shape[0]):
                data.append((ecog_data[i], labels[i]))
    return data # (Length,Channel)