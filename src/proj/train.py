from proj.model import Model
from proj.data import MyDataset
import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib as plt

DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

def train():
    model = Model(num_classes=32, pretrained=False)
    model.to(DEVICE)

    dataset = MyDataset("data/raw")
    dataset.preprocess(Path("data/processed"))
    train_dataloader = torch.utils.data.DataLoader(dataset.train_set, 64)
    
    statistics = {"loss": [], "accuracy": []}

    optimizer = torch.optim.Adam(lr=1e-3, weight_decay=0, eps=1e-8, betas=[0.9, 0.999], params=model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    for e in tqdm(range(10), desc="Training"):
        model.train()
        for audio, label in train_dataloader:
            audio, label = audio.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(audio)
            loss = criterion(prediction, label)
            accuracy = (prediction.argmax(dim=1) == label).float().mean()
            loss.backward()

            optimizer.step()

        statistics["loss"].append(loss.item())
        statistics["accuracy"].append(accuracy.item())
        print(f"\nEpoch: {e} | Loss: {loss.item()} | Accuracy: {accuracy.item()}")

        torch.save(model.state_dict(), "models/model.pth")
        #eval("models/model.pth", 64, dataset, model)

    print("Training complete")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(f"reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
