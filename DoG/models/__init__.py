from .mlp import build_mlp

MODEL_REGISTRY = {
    "mlp": build_mlp,
}

def get_model(name: str):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]

