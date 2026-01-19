import torch
import matplotlib.pyplot as plt
from pathlib import Path

def dataset_statistics(output_folder: str = "data/processed"):
    """Compute dataset statistics."""
    train_file = Path(f"{output_folder}/train/train.pt")
    test_file = Path(f"{output_folder}/test/test.pt")

    if not train_file.exists():
        print(f"Dataset not preprocessed. Call preprocess() first.")
        return

    train_data = torch.load(train_file, weights_only=True)
    test_data = torch.load(test_file, weights_only=True)

    train_labels = train_data["labels"]
    test_labels = test_data["labels"]
    train_specs = train_data["spectrograms"]

    print(f"Training samples: {len(train_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Spectrogram shape: {train_specs.shape[2:]} (Freq bins x Time frames)")

    plt.figure(figsize=(12, 5))
    unique, counts = torch.unique(train_labels, return_counts=True)
    plt.bar(unique.numpy(), counts.numpy(), color='forestgreen')
    plt.title("Training set: species distribution")
    plt.xlabel("Class label ID")
    plt.ylabel("Sample count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    unique, counts = torch.unique(test_labels, return_counts=True)
    plt.bar(unique.numpy(), counts.numpy(), color='indianred')
    plt.title("Test set: species distribution")
    plt.xlabel("Class label ID")
    plt.ylabel("Sample count")
    plt.savefig("test_label_distribution.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.imshow(train_specs[0][0].numpy(), aspect='auto', origin='lower')
    plt.title("Sample Mel-Spectrogram")
    plt.colorbar(label='dB')
    plt.savefig("sample_spectrogram.png")
    plt.close()

if __name__ == "__main__":
    dataset_statistics()