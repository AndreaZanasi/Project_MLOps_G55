from torch import nn
import torch
import torchvision.models as models

"""
Saved 1357 train (shape: torch.Size([1357, 1, 64, 1168]))
Saved 340 test (shape: torch.Size([340, 1, 64, 1168]))
"""

class Model(nn.Module):
    """Already trained ResNet34 to exctract features from audio files"""
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet34

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)

if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
