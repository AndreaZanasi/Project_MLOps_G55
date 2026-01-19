from proj.model import Model
from proj.data import MyDataset
from proj.evaluate import evaluate
import torch
import wandb
import logging
import hydra
from omegaconf import OmegaConf
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
        run: wandb.Run | None,
        dataset: MyDataset,
        batch_size: int = 32,
        epochs: int = 10,
        figures_dir: str = "reports/figures",
        model_dir: str = "models",
        model_name: str = "model.pth",
        log_wandb: bool = True
):
    train_dataloader = torch.utils.data.DataLoader(dataset.train_set, batch_size, shuffle=True)

    statistics = {"loss": [], "accuracy": []}
    best_accuracy = 0.0

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

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
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * label.size(0)
            epoch_correct += (prediction.argmax(dim=1) == label).sum().item()
            epoch_total += label.size(0)
        
        loss = epoch_loss / epoch_total
        accuracy = epoch_correct / epoch_total
        statistics["loss"].append(loss)
        statistics["accuracy"].append(accuracy)

        if log_wandb:
            run.log({"train_loss": loss, "train_accuracy" : accuracy})
        log.info(f"Epoch: {e} | Loss: {loss:.4f} | Train accuracy: {accuracy:.4f}")

        val_acc = evaluate(
            model,
            run,
            dataset,
            batch_size,
            None
        )

        if e == 0 or val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), model_name)
            log.info(
                f"New best model saved with validation accuracy: {val_acc:.4f}"
            )

    log.info("Training complete")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(f"{figures_dir}/training_statistics.png")

    if log_wandb:
        artifact = wandb.Artifact(
            name="species_recognition_model",
            type="model",
            description="A model trained to recognize species based on different animal vocalizations",
            metadata={"accuracy": best_accuracy}
        )
        artifact.add_file(model_name)
        run.log_artifact(artifact)

        run.log({"training_statistics": wandb.Image(fig)})
        run.finish()

@hydra.main(config_path="../../configs", config_name="hydra_cfg.yaml", version_base="1.1")
def main(cfg):

    if cfg.logging.log_wandb:
        run = wandb.init(
            entity="MLOps_G55",
            project="Project_MLOps_G55",
            config=OmegaConf.to_object(cfg)
        )
    else: run = None

    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    model = Model(cfg)
    model.to(DEVICE)

    dataset = MyDataset(cfg.paths.data_dir)
    dataset.preprocess(Path(cfg.paths.output_dir))

    train(
        hydra.utils.instantiate(cfg.optimizer, params=model.parameters()),
        hydra.utils.instantiate(cfg.criterion),
        model,
        run,
        dataset,
        cfg.hyperparameters.batch_size,
        cfg.hyperparameters.epochs,
        cfg.paths.figures_dir,
        cfg.paths.model_dir,
        cfg.paths.model_name,
        cfg.logging.log_wandb
    )


if __name__ == "__main__":
    main()
