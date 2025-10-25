from .dog_momentum import build_dog_momentum
from .dog_coordinate import build_dog_coordinate
from .adam import build_adam

OPTIMIZER_REGISTRY = {
    "dog_momentum": build_dog_momentum,  
    "dog": build_dog_coordinate,        
    "adam": build_adam,               
}

def get_optimizer(name: str):
    if name not in OPTIMIZER_REGISTRY:
        raise ValueError(f"Unknown optimizer '{name}'. Available: {list(OPTIMIZER_REGISTRY)}")
    return OPTIMIZER_REGISTRY[name]

