"""Model definitions.

One architecture is used for every experimental arm so that "pretrained init"
changes only the initial weights, not the capacity. ResNet-18 at 28x28 is a
deliberate choice: it has a canonical ImageNet checkpoint, so the transfer arm
is reproducible without any private data, and it is small enough that DP-SGD
per-sample gradients fit on a 6 GB consumer GPU.

Opacus cannot differentiate through BatchNorm because per-sample gradients
would need batch statistics; every BatchNorm is therefore replaced by GroupNorm
*before* the privacy engine wraps the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_CLASSES_BINARY = 2
NUM_CLASSES_CHEST = 14


def _basic_block_forward(self, x: torch.Tensor) -> torch.Tensor:
    """torchvision's BasicBlock does `out += identity`, which autograd forbids
    once Opacus' backward hook returns the tensor as a view. Add out of place."""
    identity = x
    out = self.conv1(x)
    out = self.gn1(out) if hasattr(self, "gn1") else self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.gn2(out) if hasattr(self, "gn2") else self.bn2(out)
    if self.downsample is not None:
        identity = self.downsample(x)
    out = out + identity
    out = self.relu(out)
    return out


class GroupNormResNet18(nn.Module):
    """ResNet-18 adapted to 1x28x28 input with GroupNorm in place of BatchNorm."""

    def __init__(self, num_classes: int = 2, in_channels: int = 1, groups: int = 8):
        super().__init__()
        from torchvision.models import resnet18

        try:
            net = resnet18(weights=None)
        except TypeError:  # older torchvision
            net = resnet18(pretrained=False)

        # Adapt the 7x7 stem from 3 to 1 input channel by summing the RGB filters.
        old = net.conv1.weight.data
        new = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new.weight.data = old.sum(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) / 3.0
        net.conv1 = new

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avgpool = net.avgpool
        self.fc = net.fc

        self._replace_bn_with_gn(groups)
        self._make_relu_out_of_place()
        for block in (*self.layer1, *self.layer2, *self.layer3, *self.layer4):
            block.forward = _basic_block_forward.__get__(block, type(block))
        for m in self.modules():
            if isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        self.fc = nn.Linear(self.fc.in_features, num_classes)
        self.num_classes = num_classes

    def _replace_bn_with_gn(self, groups: int) -> None:
        def swap(parent: nn.Module) -> None:
            for name, child in parent.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    ch = child.num_features
                    g = groups if ch % groups == 0 else 1
                    setattr(parent, name, nn.GroupNorm(g, ch))
                else:
                    swap(child)

        swap(self)

    def _make_relu_out_of_place(self) -> None:
        """Opacus hooks cannot backpropagate through in-place ReLU: the hook's
        output is a view that the in-place op then mutates, which autograd
        rejects. Swapping to non-in-place costs a little memory and is required
        for per-sample gradients."""
        for parent in self.modules():
            for name, child in parent.named_children():
                if isinstance(child, (nn.ReLU, nn.ReLU6)):
                    setattr(parent, name, nn.ReLU(inplace=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class SmallCNN(nn.Module):
    """Compact reference model used for the ChestMNIST scale check.

    Three conv blocks and one linear head; GroupNorm throughout so the same
    class works under Opacus without module surgery.
    """

    def __init__(self, num_classes: int = 14, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.features(x), 1))


def load_imageNet_resnet18_into(model: GroupNormResNet18) -> None:
    """Copy an ImageNet checkpoint into the adapted architecture.

    GroupNorm is initialised to weight=1, bias=0, which reproduces the identity
    scaling that the corresponding BatchNorm layers had just after training
    (their running statistics had already absorbed the scale). The 1-channel
    stem is the mean of the pretrained RGB filters.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    ref = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    ref_state = ref.state_dict()

    mapping = [
        ("layer1", "layer1"),
        ("layer2", "layer2"),
        ("layer3", "layer3"),
        ("layer4", "layer4"),
    ]
    own = model.state_dict()
    with torch.no_grad():
        for ours, theirs in mapping:
            prefix_ours = f"{ours}."
            prefix_theirs = f"{theirs}."
            for k, v in ref_state.items():
                if not k.startswith(prefix_theirs):
                    continue
                key = prefix_ours + k[len(prefix_theirs):]
                if key in own and own[key].shape == v.shape:
                    own[key].copy_(v)
        # stem conv: average the three pretrained input filters
        if "stem.0.weight" in own and "conv1.weight" in ref_state:
            w = ref_state["conv1.weight"]
            if own["stem.0.weight"].shape[0] == w.shape[0]:
                own["stem.0.weight"].copy_(w.sum(dim=1, keepdim=True) / 3.0)
        if "fc.weight" in own and own["fc.weight"].shape == ref_state["fc.weight"].shape:
            own["fc.weight"].copy_(ref_state["fc.weight"])
            own["fc.bias"].copy_(ref_state["fc.bias"])


def build_model(
    arch: str,
    num_classes: int,
    pretrained: bool = False,
    in_channels: int = 1,
) -> nn.Module:
    if arch == "resnet18":
        model = GroupNormResNet18(num_classes=num_classes, in_channels=in_channels)
        if pretrained:
            load_imageNet_resnet18_into(model)
        return model
    if arch == "smallcnn":
        if pretrained:
            raise ValueError("smallcnn has no supported pretrained checkpoint")
        return SmallCNN(num_classes=num_classes, in_channels=in_channels)
    raise ValueError(f"unknown arch: {arch}")



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
