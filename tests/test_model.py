
from proj.model import Model


class TestClass:
    def test_model_instance(self):
        """Test the Model class."""
        

        model = Model(cfg)
        assert isinstance(model, Model)