from proj.model import Model

import torch
from hydra import initialize, compose
import numpy as np
import onnxruntime as ort
import subprocess
import os

import wandb
api = wandb.Api()
artifact = api.artifact(
    "MLOps_G55/Project_MLOps_G55/species_recognition_model:v0",
    type="model",
)
artifact_dir = artifact.download()


SHAPE = (1, 64, 1168)

def export_model(path, name, weights_path):
    with initialize(config_path="../../../configs", version_base="1.1"):
        cfg = compose(config_name="model_cfg.yaml")

    model = Model(cfg)
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(2, *SHAPE)

    with torch.no_grad():
        torch.onnx.export(
            model,
            args=dummy_input,
            f=f"{path}/{name}",
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )

def visualize_model(path):
    print("Starting Netron...")
    subprocess.Popen(["netron", path])

def inference(ort_session: ort.InferenceSession, audio):
    input_names = [i.name for i in ort_session.get_inputs()]
    output_names = [o.name for o in ort_session.get_outputs()]
    batch = {input_names[0]: audio.astype(np.float32)}
    output = ort_session.run(output_names, batch)
    return output[0]

if __name__ == "__main__":
    print(os.listdir(artifact_dir))
    checkpoint_path = os.path.join(artifact_dir, "model.pth")
    export_model("models", "model.onnx", checkpoint_path)
    visualize_model("models/model.onnx")