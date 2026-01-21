import os
import sys
import time
import torch
import wandb
from hydra import initialize, compose
from src.proj.lightning_model import LightningAudioClassifier

sys.path.append(os.getcwd())


def load_model(artifact_path):
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))

    artifact = api.artifact(artifact_path)
    logdir = "models/registry_checkpoint"
    artifact.download(root=logdir)

    with initialize(version_base="1.1", config_path="../../configs"):
        cfg = compose(config_name="model_cfg.yaml")

    checkpoint_path = os.path.join(logdir, "model.ckpt")

    model = LightningAudioClassifier.load_from_checkpoint(checkpoint_path=checkpoint_path, cfg=cfg)
    model.eval()
    return model


def test_model_speed():
    model_name = os.getenv("MODEL_NAME", "MLOps_G55/marine_mammal_registry/species_classifier:staging")
    model = load_model(model_name)

    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(torch.rand(1, 1, 28, 28))
    end = time.time()

    speed = end - start
    print(f"Time: {speed:.4f}s")
    assert speed < 10.0


if __name__ == "__main__":
    test_model_speed()
