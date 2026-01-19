from torch.utils.data import Dataset

from proj.data import MyDataset


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
