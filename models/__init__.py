from .mlp import build_mlp
from .cnn import build_cnn
from .resnet import build_resnet

from .gpt import build_gpt_small, build_gpt2_preset

MODEL_REGISTRY = {
    "mlp": build_mlp,
    "cnn": build_cnn,
    "resnet": build_resnet,
    "gpt": build_gpt_small,   
    "gpt2": build_gpt2_preset,
}

def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]



def get_model(name: str):
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")

__all__ = [
    "get_model",
    "build_mlp",
    "build_cnn",
    "build_resnet",
    "build_gpt_small",
    "build_gpt2_preset",
]

