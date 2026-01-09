from proj.model import Model
from proj.data import MyDataset
import torch
import logging
import hydra
from omegaconf import OmegaConf
from hydra import initialize, compose
from tqdm import tqdm
from pathlib import Path
import matplotlib as plt

log = logging.getLogger(__name__)

DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

def train(
        model,
        optimizer, 
        criterion,
        batch_size: int = 32,
        epochs: int = 10,
        data_dir: str = "data/raw",
        output_dir: str = "data/processed",
        figures_dir: str = "reports/figures",
        model_name: str = "models/model.pt"
):
    dataset = MyDataset(data_dir)
    dataset.preprocess(Path(output_dir))
    train_dataloader = torch.utils.data.DataLoader(dataset.train_set, batch_size)
    
    statistics = {"loss": [], "accuracy": []}

    for e in tqdm(range(epochs), desc="Training"):
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
        log.info(f"\nEpoch: {e} | Loss: {loss.item()} | Accuracy: {accuracy.item()}")

        torch.save(model.state_dict(), model_name)

    log.info("Training complete")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(f"{figures_dir}/training_statistics.png")

def main():
    with initialize(config_path="../../configs", version_base="1.1"):
        train_cfg = compose(config_name="train_cfg.yaml")
        model_cfg = compose(config_name="model_cfg.yaml")

    log.info("Training Configuration:")
    log.info(OmegaConf.to_yaml(train_cfg))
    log.info("\nModel Configuration:")
    log.info(OmegaConf.to_yaml(model_cfg))

    model = Model(model_cfg)
    model.to(DEVICE)

    train(
        model,
        hydra.utils.instantiate(train_cfg.optimizer, params=model.parameters()),
        hydra.utils.instantiate(train_cfg.criterion),
        train_cfg.hyperparameters.batch_size,
        train_cfg.hyperparameters.epochs,
        train_cfg.paths.data_dir,
        train_cfg.paths.output_dir,
        train_cfg.paths.figures_dir,
        train_cfg.paths.model_name
    )

if __name__ == "__main__":
    main()
