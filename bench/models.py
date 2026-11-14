import torch.nn as nn
import torch.nn.functional as F


def _cbr(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class BNCNN(nn.Module):
    """CNN with 5 conv layers and BatchNorm, shared by Fashion-MNIST (1 channel) and CIFAR (3 channels)."""

    def __init__(self, in_ch, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            *_cbr(in_ch, 64), *_cbr(64, 64), nn.MaxPool2d(2),
            *_cbr(64, 128), *_cbr(128, 128), nn.MaxPool2d(2),
            *_cbr(128, 256), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(256, num_classes)
        # Same conv initialization as PlainCNN, so that the ablation differs only in BatchNorm.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x):
        # mean() instead of AdaptiveAvgPool2d: the backward pass of AdaptiveAvgPool2d on CUDA
        # is not deterministic.
        return self.fc(self.features(x).mean((2, 3)))


class PlainCNN(nn.Module):
    """Identical to BNCNN but WITHOUT BatchNorm (convs have biases); used for the BN ablation."""

    def __init__(self, in_ch, num_classes=10):
        super().__init__()

        def cr(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1, bias=True), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(
            *cr(in_ch, 64), *cr(64, 64), nn.MaxPool2d(2),
            *cr(64, 128), *cr(128, 128), nn.MaxPool2d(2),
            *cr(128, 256), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(256, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.fc(self.features(x).mean((2, 3)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    """ResNet-18 for 32x32 images (3x3 stem conv, no max-pooling)."""

    def __init__(self, in_ch=3, num_classes=10):
        super().__init__()
        self.cin = 64
        self.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    def _make(self, cout, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.cin, cout, s))
            self.cin = cout
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = out.mean((2, 3))
        return self.fc(out)


def build_model(name, in_ch, num_classes=10):
    if name == "bncnn":
        return BNCNN(in_ch, num_classes)
    if name == "resnet18":
        return ResNet18(in_ch, num_classes)
    if name == "plaincnn":
        return PlainCNN(in_ch, num_classes)
    raise ValueError(f"unsupported model: {name}")
