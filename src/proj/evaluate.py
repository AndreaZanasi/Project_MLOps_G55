import torch
import logging
from hydra import initialize, compose
from proj.model import Model
from proj.data import MyDataset

log = logging.getLogger(__name__)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def evaluate(
    model: Model,
    dataset: MyDataset,
    batch_size: int,
    model_checkpoint: str | None = None
):  
    if model_checkpoint:
        model.load_state_dict(torch.load(model_checkpoint, weights_only=False))
        
    test_dataloader = torch.utils.data.DataLoader(dataset.test_set, batch_size, shuffle=True)

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for audio, label in test_dataloader:
            audio, label = audio.to(DEVICE), label.to(DEVICE)

            prediction = model(audio)
            correct += (prediction.argmax(dim=1) == label).float().sum().item()
            total += label.size(0)

    log.info(f"Model eval accuracy: {(correct / total):.4f}")


def main():
    with initialize(config_path="../../configs", version_base="1.1"):
        train_cfg = compose(config_name="train_cfg.yaml")
        model_cfg = compose(config_name="model_cfg.yaml")

    model = Model(model_cfg)
    model.to(DEVICE)

    ds = MyDataset(train_cfg.paths.data_dir)
    ds.preprocess(train_cfg.paths.output_dir)

    evaluate(
        model,
        ds.test_set,
        train_cfg.hyperparameters.batch_size,
        train_cfg.paths.model_name
    )


if __name__ == "__main__":
    main()
