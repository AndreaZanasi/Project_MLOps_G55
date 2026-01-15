
from proj.model import Model


class TestClass:
    def test_model_instance(self):
        """Test the Model class."""
        cfg = type("cfg", (), {})()  # Create a simple empty config object
        cfg.parameters = type("parameters", (), {})()
        cfg.parameters.features = [1, 64]
        cfg.parameters.kernel_sizes = [(7, 7)]
        cfg.parameters.strides = [(2, 2)]
        cfg.parameters.paddings = [(3, 3)]
        cfg.setup = type("setup", (), {})()
        cfg.setup.pretrained = False
        cfg.setup.num_classes = 32

        model = Model(cfg)
        assert isinstance(model, Model)