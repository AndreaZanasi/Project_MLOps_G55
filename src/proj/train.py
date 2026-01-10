from proj.model import Model
from proj.data import MyDataset
from proj.evaluate import evaluate
import torch
import logging
import hydra
from omegaconf import OmegaConf
from hydra import initialize, compose
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

def train(
        optimizer, 
        criterion,
        model: Model,
        batch_size: int = 32,
        epochs: int = 10,
        data_dir: str = "data/raw",
        output_dir: str = "data/processed",
        figures_dir: str = "reports/figures",
        model_name: str = "models/model.pt"
):
    dataset = MyDataset(data_dir)
    dataset.preprocess(Path(output_dir))
    train_dataloader = torch.utils.data.DataLoader(dataset.train_set, batch_size, shuffle=True)
    
    statistics = {"loss": [], "accuracy": []}

    for e in tqdm(range(epochs), desc="Training"):
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for audio, label in train_dataloader:
            audio, label = audio.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(audio)
            loss = criterion(prediction, label)
            accuracy = (prediction.argmax(dim=1) == label).float().mean()
            loss.backward()

            optimizer.step()
        
        statistics["loss"].append(epoch_loss / epoch_total)
        statistics["accuracy"].append(epoch_correct / epoch_total)
        log.info(f"Epoch: {e} | Loss: {(epoch_loss / epoch_total):.4f} | Accuracy: {(epoch_correct / epoch_total):.4f}")

        torch.save(model.state_dict(), model_name)

        evaluate(
            model,
            dataset.test_set,
            batch_size,
            None,
            log
        )

    log.info("Training complete")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(f"{figures_dir}/training_statistics.png")

@hydra.main(config_path="../../configs", config_name="hydra_cfg.yaml", version_base="1.1")
def main(cfg):
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    model = Model(cfg)
    model.to(DEVICE)

    train(
        hydra.utils.instantiate(cfg.optimizer, params=model.parameters()),
        hydra.utils.instantiate(cfg.criterion),
        model,
        cfg.hyperparameters.batch_size,
        cfg.hyperparameters.epochs,
        cfg.paths.data_dir,
        cfg.paths.output_dir,
        cfg.paths.figures_dir,
        cfg.paths.model_name
    )

if __name__ == "__main__":
    main()
