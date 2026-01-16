import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchvision.models as models


class LightningAudioClassifier(L.LightningModule):
    """PyTorch Lightning module for audio classification."""
    
    def __init__(self, cfg, learning_rate: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.learning_rate = learning_rate
        
        # Build model
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
        
        # For metrics tracking (optional - Lightning handles this automatically)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
    def forward(self, x):
        """Forward pass through the model."""
        return self.resnet(x)
    
    def training_step(self, batch, batch_idx):
        """Training step - replaces your manual training loop."""
        audio, labels = batch
        outputs = self(audio)
        loss = F.cross_entropy(outputs, labels)
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        # Lightning automatically handles logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step - automatic validation loop."""
        audio, labels = batch
        outputs = self(audio)
        loss = F.cross_entropy(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step - for final evaluation."""
        audio, labels = batch
        outputs = self(audio)
        loss = F.cross_entropy(outputs, labels)
        
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers - Lightning handles the optimization loop."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }