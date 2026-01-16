
from proj.model import Model
from hydra import compose, initialize
import torch
from tests import OUT_SHAPE


class TestClass:
    def test_model_instance(self):
        """Test the Model class."""
        with initialize(config_path="../configs", version_base="1.1"):
            model_cfg = compose(config_name="model_cfg.yaml")

        model = Model(model_cfg)
        assert isinstance(model, Model)

        x = torch.rand(4, 1, 64, 1168)
        output = model(x)
        assert output.shape == OUT_SHAPE, f"Expected shape: {OUT_SHAPE}"