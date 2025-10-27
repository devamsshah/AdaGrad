import os

# Map logical dataset names -> on-disk subfolders under data_root
_DATASET_SUBDIR = {
    "uci_har": "UCI_HAR/UCI_HAR_Dataset",
    "cifar10": "CIFAR10",
    "mnist":   "MNIST",
}

def resolve_data_dir(dataset: str, data_root: str, data_dir_override: str | None) -> str:
    if data_dir_override:
        return data_dir_override
    sub = _DATASET_SUBDIR.get(dataset, dataset)  # fallback: use dataset name
    return os.path.join(data_root, sub)

