from torch import nn
import torch
import torchvision.models as models

"""
Saved 1357 train (shape: torch.Size([1357, 1, 64, 1168]))
Saved 340 test (shape: torch.Size([340, 1, 64, 1168]))

There are 32 species
"""

class Model(nn.Module):
    """Already trained ResNet34 to exctract features from audio files"""
    def __init__(self, num_classes: int = 32, pretrained: bool = False) -> None:
        super().__init__()
        weights = "ResNet34_Weights.DEFAULT" if pretrained else None
        self.resnet = models.resnet34(weights=weights)

        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

if __name__ == "__main__":

    model = Model(num_classes=32, pretrained=False)
    x = torch.rand(4, 1, 64, 1168)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
