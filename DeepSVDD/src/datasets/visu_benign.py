import torch
from torch.utils.data import Dataset

class VisuTrainDataset(Dataset):
    def __init__(self, path):
        self.data = torch.load(path).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], 0, idx  # I had to add index to be compatible with ae_trainer.py file

# label = 0 for benign


class VisuTestDataset(Dataset):
    def __init__(self, path_embeddings, path_labels):
        self.data = torch.load(path_embeddings).float()
        self.labels = torch.load(path_labels).long()

        assert len(self.data) == len(self.labels), "Mismatch between test data and labels!"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], idx
