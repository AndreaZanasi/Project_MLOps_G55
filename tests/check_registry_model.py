import os
import wandb
import torch
from hydra import initialize, compose
import sys
sys.path.append(os.getcwd())
from src.proj.model import Model

def check_model():
    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY"))
    
    entity = "MLOps_G55"
    registry_path = f"{entity}/marine_mammal_registry/species_classifier:staging"
    
    artifact = api.artifact(registry_path)
    download_path = artifact.download()
    
    with initialize(version_base="1.1", config_path="../configs"):
        cfg = compose(config_name="model_cfg.yaml")
    
    model = Model(cfg=cfg)
    model.load_state_dict(torch.load(os.path.join(download_path, "model.pth")))
    model.eval()
    
    dummy_input = torch.rand(1, 1, 64, 1168)
    output = model(dummy_input)
    print(f"Model check successful. Output shape: {output.shape}")

if __name__ == "__main__":
    check_model()