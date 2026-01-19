from proj.train import train
from proj.model import Model
import torch
import pytest
from hydra import compose, initialize
from tests import SHAPE


@pytest.fixture
def setup_train():
    """Setup for training tests."""
    with initialize(config_path="../configs", version_base="1.1"):
        cfg = compose(config_name="hydra_cfg.yaml")
    
    model = Model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    return model, optimizer, criterion

def test_train_smoke(setup_train, tmp_path):
    """Smoke test: verify training runs without crashing."""
    model, optimizer, criterion = setup_train
    
    audio = torch.randn(1, *SHAPE)
    label = torch.tensor([0])
    dataset = torch.utils.data.TensorDataset(audio, label)
    
    from proj.data import MyDataset
    mock_dataset = MyDataset("data/test/raw")
    mock_dataset.preprocess("data/test/processed")
    mock_dataset.train_set = dataset
    mock_dataset.test_set = dataset
    
    train(
        optimizer,
        criterion,
        model,
        mock_dataset,
        batch_size=1,
        epochs=1,
        figures_dir=str(tmp_path / "figures"),
        model_dir=str(tmp_path / "models"),
        model_name="test_model.pth",
        log_wandb=False
    )