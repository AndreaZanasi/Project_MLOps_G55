import re
from torch.utils.data import Dataset

from proj.data import MyDataset
import pytest


class TestClass:
    def test_my_dataset(self):
        """Test the MyDataset class."""
        dataset = MyDataset("data/raw")
        assert isinstance(dataset, Dataset)

    def test_my_dataset_methods(self):
        """Test the methods of MyDataset class."""
        dataset = MyDataset("data/raw")
        assert hasattr(dataset, "preprocess")
        assert hasattr(dataset, "spec_to_tensors")
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")

    def test_error_dataset_not_preprocessed(self):
        """Test error when accessing item before preprocessing."""
        dataset = MyDataset("data/raw")
        msg = "Dataset not preprocessed. Call preprocess() first."
        with pytest.raises(ValueError, match=re.escape(msg)):
            _ = dataset[0]
