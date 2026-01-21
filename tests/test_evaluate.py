import pytest
import torch
from unittest.mock import Mock
from proj.evaluate import evaluate
from proj.lightning_model import LightningAudioClassifier as Model
from proj.data import MyDataset
from tests import N_CLASSES, SHAPE


class TestEvaluate:
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset for testing."""
        dataset = Mock(spec=MyDataset)
        # Create synthetic test data
        test_spectrograms = torch.randn(10, *SHAPE)
        test_labels = torch.randint(0, N_CLASSES, (10,))
        test_dataset = torch.utils.data.TensorDataset(test_spectrograms, test_labels)
        dataset.test_set = test_dataset
        return dataset

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model_cfg = Mock()
        model_cfg.num_classes = N_CLASSES
        model_cfg.input_shape = SHAPE
        model = Model(model_cfg)
        model.eval()
        return model

    def test_evaluate_accuracy_calculation(self, mock_dataset):
        """Test accuracy calculation with a deterministic model."""

        # Create a simple model that always predicts class 0
        class DeterministicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                batch_size = x.size(0)
                # Return logits that always select class 0
                logits = torch.zeros(batch_size, N_CLASSES)
                logits[:, 0] = 100.0  # High value for class 0
                return logits

        model = DeterministicModel()

        # Create dataset where all labels are 0
        test_spectrograms = torch.randn(10, *SHAPE)
        test_labels = torch.zeros(10, dtype=torch.long)
        mock_dataset.test_set = torch.utils.data.TensorDataset(test_spectrograms, test_labels)

        accuracy = evaluate(
            model=model,
            run=None,
            dataset=mock_dataset,
            batch_size=4,
            log_wandb=False,
            model_checkpoint=None,
        )

        # All predictions should be correct
        assert accuracy == 1.0

    def test_evaluate_zero_accuracy(self, mock_dataset):
        """Test evaluation with a model that always predicts wrong."""

        # Create a model that always predicts class 1
        class WrongModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                batch_size = x.size(0)
                logits = torch.zeros(batch_size, N_CLASSES)
                logits[:, 1] = 100.0  # High value for class 1
                return logits

        model = WrongModel()

        # Create dataset where all labels are 0 (not 1)
        test_spectrograms = torch.randn(10, *SHAPE)
        test_labels = torch.zeros(10, dtype=torch.long)
        mock_dataset.test_set = torch.utils.data.TensorDataset(test_spectrograms, test_labels)

        accuracy = evaluate(
            model=model,
            run=None,
            dataset=mock_dataset,
            batch_size=4,
            log_wandb=False,
            model_checkpoint=None,
        )

        # All predictions should be wrong
        assert accuracy == 0.0
