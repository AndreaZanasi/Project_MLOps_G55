from torch import nn
import torch
import hydra
import torchvision.models as models

"""
Saved 1357 train (shape: torch.Size([1357, 1, 64, 1168]))
Saved 340 test (shape: torch.Size([340, 1, 64, 1168]))

There are 32 species
"""

class Model(nn.Module):
    """Already trained ResNet34 to exctract features from audio files"""
    def __init__(self, cfg) -> None:
        super().__init__()

        features = cfg.parameters.features
        kernel_sizes = cfg.parameters.kernel_sizes
        strides = cfg.parameters.strides
        paddings = cfg.parameters.paddings

        weights = "ResNet34_Weights.DEFAULT" if cfg.setup.pretrained else None
        self.resnet = models.resnet34(weights=weights)

        self.resnet.conv1 = nn.Conv2d(
            features[0], 
            features[1], 
            kernel_size=kernel_sizes[0], 
            stride=strides[0], 
            padding=paddings[0]
        )

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, cfg.setup.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

@hydra.main(config_path="../../configs", config_name="hydra_cfg.yaml", version_base="1.1")
def main(model_cfg):
    model = Model(model_cfg)
    x = torch.rand(4, 1, 64, 1168)
    
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    main()