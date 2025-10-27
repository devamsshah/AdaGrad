from .mlp import build_mlp
from .cnn import build_cnn
from .resnet import build_resnet

MODEL_REGISTRY = {
    "mlp": build_mlp,
    "cnn": build_cnn,
    "resnet": build_resnet,
}

def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]

