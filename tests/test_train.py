from proj.train_lightning import train_lightning
import torch
import pytest
from hydra import compose, initialize
from unittest.mock import MagicMock
from torch.utils.data import DataLoader
from tests import SHAPE

@pytest.fixture
def setup_configs(tmp_path):
    """Setup for training tests."""
    with initialize(config_path="../configs", version_base="1.1"):
        train_cfg = compose(config_name="hydra_cfg.yaml", overrides=[
            "logging.log_wandb=False",
            f"paths.model_dir={str(tmp_path)}"
        ])
        model_cfg = compose(config_name="model_cfg.yaml")
    return train_cfg, model_cfg

def test_train_smoke(setup_configs):
    """Smoke test: verify training runs without crashing."""
    train_cfg, model_cfg = setup_configs
    
    audio = torch.randn(2, *SHAPE)
    label = torch.randint(0, 5, (2,))
    dataset = torch.utils.data.TensorDataset(audio, label)
    
    mock_dm = MagicMock()
    loader = DataLoader(dataset, batch_size=2)
    mock_dm.train_dataloader.return_value = loader
    mock_dm.val_dataloader.return_value = loader
    mock_dm.test_dataloader.return_value = loader
    
    train_lightning(
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        max_epochs=1,
        data_module=mock_dm,
        accelerator="cpu",
        devices=1,
        log_wandb=False
    )