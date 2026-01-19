import lightning as L
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from proj.data import MyDataset


class AudioDataModule(L.LightningDataModule):
    """Lightning Data Module for audio classification."""

    def __init__(
        self,
        data_dir: str = "data/raw",
        output_dir: str = "data/processed",
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Download and preprocess data (called only on main process)."""
        # This ensures data is only downloaded/processed once
        dataset = MyDataset(self.data_dir)
        dataset.preprocess(self.output_dir)

    def setup(self, stage=None):
        """Setup datasets for each stage."""
        if self.dataset is None:
            self.dataset = MyDataset(self.data_dir)
            self.dataset.preprocess(self.output_dir)

        if stage == "fit" or stage is None:
            # Split training data into train/val
            train_size = int((1 - self.val_split) * len(self.dataset.train_set))
            val_size = len(self.dataset.train_set) - train_size

            self.train_dataset, self.val_dataset = random_split(
                self.dataset.train_set, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.dataset.test_set

    def train_dataloader(self):
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False
        )
