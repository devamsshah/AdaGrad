import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2, num_classes: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, x):
        return self.net(x)

def build_mlp(input_dim: int, num_classes: int, args) -> nn.Module:
    return MLP(in_dim=input_dim, hidden=args.hidden, dropout=args.dropout, num_classes=num_classes)

