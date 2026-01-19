from torch.utils.data import Dataset

from proj.data import MyDataset

from pathlib import Path

from tests import TRAIN_LEN, TEST_LEN, SHAPE, N_CLASSES


class TestClass:
    def test_my_dataset(self):
        """Test the MyDataset class."""
        dataset = MyDataset("data/raw")
        assert isinstance(dataset, Dataset)

        dataset.preprocess(Path("data/processed"))
        assert len(dataset.train_set) == TRAIN_LEN, f"Expected train set length: {TRAIN_LEN}"
        assert len(dataset.test_set) == TEST_LEN, f"Expected test set length: {TEST_LEN}"

        for ds in [dataset.train_set, dataset.test_set]:
            for audio, labels in ds:
                assert audio.shape == SHAPE, f"Expected shape: {SHAPE}"
                assert labels in range(N_CLASSES), f"Expected number of classes: {N_CLASSES}"


    def test_my_dataset_methods(self):
        """Test the methods of MyDataset class."""
        dataset = MyDataset("data/raw")
        assert hasattr(dataset, "preprocess")
        assert hasattr(dataset, "spec_to_tensors")
        assert hasattr(dataset, "__len__")
        assert hasattr(dataset, "__getitem__")