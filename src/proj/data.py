from pathlib import Path

import typer
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset
from datasets import load_dataset


class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.train_set = None
        self.test_set = None

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        train_folder = output_folder / "train"
        test_folder = output_folder / "test"

        train_file = train_folder / "train.pt"
        test_file = test_folder / "test.pt"

        if train_file.exists() and test_file.exists():
            print(f"Preprocessed data already exists in {output_folder}, loading from disk...")
            train_data = torch.load(train_file, weights_only=True)
            test_data = torch.load(test_file, weights_only=True)
            self.train_set = TensorDataset(train_data["spectrograms"], train_data["labels"])
            self.test_set = TensorDataset(test_data["spectrograms"], test_data["labels"])
            print(f"Loaded {len(self.train_set)} train samples and {len(self.test_set)} test samples")
            return

        train_folder.mkdir(parents=True, exist_ok=True)
        test_folder.mkdir(parents=True, exist_ok=True)

        train_ds, test_ds = self.make_dataset()
        self.train_set = self.spec_to_tensors(train_ds, train_folder, "train")
        self.test_set = self.spec_to_tensors(test_ds, test_folder, "test")

    def spec_to_tensors(self, dataset, save_folder: Path, split: str) -> None:
        target_sr = 32000
        n_mels = 64
        n_fft = 1024
        hop_length = 320

        #This value was computed based on the average length of the audio files
        target_length = 1168

        spectrograms = []
        labels = []

        for idx, sample in enumerate(dataset):
            audio = sample["audio"]
            waveform = np.array(audio["array"], dtype=np.float32)
            original_sr = audio["sampling_rate"]

            if original_sr != target_sr:
                waveform = librosa.resample(waveform, orig_sr=original_sr, target_sr=target_sr)

                mel_spec = librosa.feature.melspectrogram(
                    y=waveform,
                    sr=target_sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels,
                    fmin=50,
                    fmax=14000,
                )

            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            #Pad/Truncate data since every file must be of the same length
            if mel_spec_db.shape[1] < target_length:
                pad_width = target_length - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
            else: 
                mel_spec_db = mel_spec_db[:, :target_length]

            tensor_data = torch.from_numpy(mel_spec_db).float().unsqueeze(0)

            spectrograms.append(tensor_data)
            labels.append(sample.get("label", 0))

        spectrograms = torch.stack(spectrograms, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)

        torch.save({"spectrograms": spectrograms, "labels": labels}, f"{save_folder}/{split}.pt")
        print(f"Saved {len(dataset)} spectrograms to {save_folder} (shape: {spectrograms.shape})")

        tensor_dataset = TensorDataset(spectrograms, labels)
        return tensor_dataset


    def make_dataset(self) -> None:
        """Download all animal sound datasets across all configs and splits."""

        dataset = "confit/wmms-parquet"
        save_path = f"{self.data_path}"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        try:
            train_ds = load_dataset(dataset, split="train")
            test_ds = load_dataset(dataset, split="test")

            train_ds.save_to_disk(save_path)
            test_ds.save_to_disk(save_path)

            print(f"Successfully downloaded dataset to {save_path}")
        except Exception as e:
            print(f"Failed to download {dataset}: {e}")

        return train_ds, test_ds


def preprocess(data_path: Path, output_folder: Path) -> None:
    """Preprocess the raw data and save it to the output folder."""
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)