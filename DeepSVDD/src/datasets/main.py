from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .visu_benign import VisuTestDataset, VisuTrainDataset
from torch.utils.data import DataLoader


class CustomDeepSVDDDataset:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

    def loaders(self, batch_size, shuffle_train=True, num_workers=2):
        train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, test_loader


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'visu')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'visu':
        train_set = VisuTrainDataset(path="/src/datasets/embeddings/benign_visu_training.pt")
        test_set = VisuTestDataset(path_embeddings="/src/datasets/embeddings/test_tensors.pt",
                                   path_labels="/src/datasets/embeddings/test_labels.pt")
        dataset = CustomDeepSVDDDataset(train_set, test_set)


    return dataset
