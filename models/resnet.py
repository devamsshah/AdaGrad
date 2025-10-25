import torch
from torch import nn

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

class ResNet(nn.Module):
    """
    Minimal ResNet supporting depths {18, 34} with BasicBlock.
    Uses AdaptiveAvgPool so no fixed input size required.
    """
    def __init__(self, block, layers, in_ch=3, base_width=64, num_classes=10, dropout=0.0):
        super().__init__()
        self.inplanes = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(block, base_width,   layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_width*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width*8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(base_width*8*block.expansion, num_classes),
        )

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return self.head(x)

_RESNET_CFG = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
}

def build_resnet(input_dim, num_classes: int, args) -> nn.Module:
    """
    input_dim:
      - tuple/list (C,H,W) preferred; otherwise falls back to args.in_channels.
    args:
      - --resnet-depth {18,34} (default 18)
      - --resnet-width  (base width, default 64)
      - --dropout       (already global)
      - optionally --in-channels (fallback)
    """
    depth = int(getattr(args, "resnet_depth", 18))
    if depth not in _RESNET_CFG:
        raise ValueError(f"Unsupported ResNet depth {depth}. Choose from {list(_RESNET_CFG)}")

    if isinstance(input_dim, (tuple, list)) and len(input_dim) >= 1:
        in_ch = int(input_dim[0])
    else:
        in_ch = int(getattr(args, "in_channels", 3))

    base_w = int(getattr(args, "resnet_width", 64))
    return ResNet(BasicBlock, _RESNET_CFG[depth],
                  in_ch=in_ch, base_width=base_w,
                  num_classes=num_classes, dropout=getattr(args, "dropout", 0.0))

