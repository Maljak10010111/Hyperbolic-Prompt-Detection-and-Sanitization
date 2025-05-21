import importlib
from torch.utils.data import DataLoader
from .datasetsEnum import DatasetName

DATASET_CLASS_MAP = {
    DatasetName.I2P: ("HySAC.hysac.dataset.i2p", "I2P"),
    DatasetName.MSCOCO: ("HySAC.hysac.dataset.mscoco", "MSCOCO"),  # Adjust if needed
    DatasetName.MMA: ("HySAC.hysac.dataset.mma", "MMA"),
}


def get_dataloader_and_dataset(
    dataset_name: DatasetName,
    dataset_args=None,
    batch_size=32,
    shuffle=False,
    num_workers=4,
):

    module_path, class_name = DATASET_CLASS_MAP[dataset_name]

    try:
        module = importlib.import_module(module_path)
        dataset_class = getattr(module, class_name)
    except Exception as e:
        raise ValueError(
            f"Could not load dataset class '{class_name}' from module '{module_path}'"
        ) from e

    dataset_args = dataset_args or {}
    dataset = dataset_class(**dataset_args)

    return (
        DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        ),
        dataset,
    )
