import sys
from sympy import compose
import wandb
import os
import time
import torch
from hydra import initialize, compose
sys.path.append(os.getcwd())
from src.proj.model import Model

def load_model(artifact_path):

    api = wandb.Api(api_key=os.getenv("WANDB_API_KEY")) 
    
    artifact = api.artifact(artifact_path)
    logdir = "models/registry_checkpoint"
    artifact.download(root=logdir)
    with initialize(version_base="1.1", config_path="../../configs"):
        cfg = compose(config_name="model_cfg.yaml")
    model = Model(cfg=cfg) 

    file_name = artifact.files()[0].name
    checkpoint_path = f"{logdir}/{file_name}"
    
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict)
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
    
    assert end - start < 1, f"Model too slow: {end - start:.4f}s"

if __name__ == "__main__":
    test_model_speed()