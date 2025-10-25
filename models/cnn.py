import torch
from torch import nn

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class SimpleCNN(nn.Module):
    """
    A compact CNN that works for arbitrary image sizes/channels.
    Uses AdaptiveAvgPool, so no need to know HxW at build time.
    """
    def __init__(self, in_ch: int, num_classes: int, width: int = 64, dropout: float = 0.0):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 4

        self.features = nn.Sequential(
            ConvBNReLU(in_ch, c1, k=3, s=1, p=1),
            ConvBNReLU(c1, c1, k=3, s=1, p=1),
            nn.MaxPool2d(2),                  # /2

            ConvBNReLU(c1, c2, k=3, s=1, p=1),
            ConvBNReLU(c2, c2, k=3, s=1, p=1),
            nn.MaxPool2d(2),                  # /4

            ConvBNReLU(c2, c3, k=3, s=1, p=1),
            ConvBNReLU(c3, c3, k=3, s=1, p=1),
            nn.AdaptiveAvgPool2d(1),          # -> (B, c3, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c3, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def build_cnn(input_dim, num_classes: int, args) -> nn.Module:
    """
    input_dim:
      - tuple/list (C, H, W) for image datasets (preferred)
      - int for tabular; if so, we fallback to args.in_channels (default 1)
    args:
      - --cnn-width (default 64)
      - --dropout (already in your global args)
      - optionally --in-channels for non-image metas
    """
    if isinstance(input_dim, (tuple, list)) and len(input_dim) >= 1:
        in_ch = int(input_dim[0])
    else:
        in_ch = int(getattr(args, "in_channels", 1))
    width = int(getattr(args, "cnn_width", 64))
    return SimpleCNN(in_ch=in_ch, num_classes=num_classes, width=width, dropout=args.dropout)

