from .uci_har import load_and_make_loaders as _uci_har

DATASET_REGISTRY = {
    "uci_har": _uci_har,  # returns (train_loader, test_loader, meta_dict)
}

def get_dataset(name: str):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]

