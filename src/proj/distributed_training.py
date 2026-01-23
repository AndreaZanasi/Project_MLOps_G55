from proj.model import Model
from proj.data import MyDataset
from argparse import ArgumentParser
from pathlib import Path
import hydra
import wandb
import torch
import logging
from omegaconf import OmegaConf
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt

SEED = 42
log = logging.getLogger(__name__)


def train(
    optimizer,
    criterion,
    device,
    model: Model,
    run: wandb.Run | None,
    dataloader: DataLoader,
    epochs: int = 10,
    figures_dir: str = "reports/figures",
    model_dir: str = "models",
    model_name: str = "model.pth",
    log_wandb: bool = True,
):
    statistics = {"loss": [], "accuracy": []}

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    for e in tqdm(range(epochs), desc="Training"):
        model.train()
        dist.barrier()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for audio, label in dataloader:
            audio, label = audio.to(device), label.to(device)

            optimizer.zero_grad()

            prediction = model(audio)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * label.size(0)
            epoch_correct += (prediction.argmax(dim=1) == label).sum().item()
            epoch_total += label.size(0)

        stats_tensor = torch.tensor([epoch_loss, epoch_correct, epoch_total], device=device, dtype=torch.float64)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        global_loss = stats_tensor[0].item() / stats_tensor[2].item()
        global_accuracy = stats_tensor[1].item() / stats_tensor[2].item()

        statistics["loss"].append(global_loss)
        statistics["accuracy"].append(global_accuracy)

        if log_wandb:
            run.log({"train_loss": loss, "train_accuracy": global_accuracy})
        log.info(f"Epoch: {e} | Loss: {loss:.4f} | Train accuracy: {global_accuracy:.4f}")
        torch.save(model.state_dict(), model_name)

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
        )
        artifact.add_file(model_name)
        run.log_artifact(artifact)

        run.log({"training_statistics": wandb.Image(fig)})
        run.finish()


@hydra.main(config_path="../../configs", config_name="hydra_cfg.yaml", version_base="1.1")
def main(cfg):
    """Script for distributed data training"""

    if cfg.logging.log_wandb:
        run = wandb.init(entity="MLOps_G55", project="Project_MLOps_G55", config=OmegaConf.to_object(cfg))
    else:
        run = None

    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    parser = ArgumentParser("DDP")
    parser.add_argument("--local_rank", type=int, default=-1, metavar="N", help="Local process rank.")
    parser.add_argument("--n_workers", type=int, default=4, help="To make the dataloader distributed")
    args = parser.parse_args()

    args.is_master = args.local_rank == 0
    args.device = torch.cuda.device(args.local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)
    torch.cuda.manual_seed_all(SEED)

    model = Model(cfg)
    model = model.to(args.device)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    dataset = MyDataset(cfg.paths.data_dir)
    dataset.preprocess(cfg.paths.output_dir)

    sampler = DistributedSampler(dataset.train_set)
    dataloader = DataLoader(
        dataset=dataset.train_set,
        sampler=sampler,
        batch_size=cfg.hyperparameters.batch_size,
        num_workers=args.n_workers,
    )

    train(
        hydra.utils.instantiate(cfg.optimizer, params=model.parameters()),
        hydra.utils.instantiate(cfg.criterion),
        args.device,
        model,
        run,
        dataloader,
        cfg.hyperparameters.epochs,
        cfg.paths.figures_dir,
        cfg.paths.model_dir,
        cfg.paths.model_name,
        cfg.logging.log_wandb,
    )
