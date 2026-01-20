from proj.model import Model

import torch
from hydra import initialize, compose
import numpy as np
import onnxruntime as ort
import subprocess

SHAPE = (1, 64, 1168)

def export_model(path, name):
    with initialize(config_path="../../../configs", version_base="1.1"):
        cfg = compose(config_name="model_cfg.yaml")

    model = Model(cfg)
    dummy_input = torch.randn(2, *SHAPE)

    model.eval()

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

def inference(path):
    ort_session = ort.InferenceSession(path)
    input_names = [i.name for i in ort_session.get_inputs()]
    output_names = [o.name for o in ort_session.get_outputs()]
    batch = {input_names[0]: np.random.randn(2, *SHAPE).astype(np.float32)}
    output = ort_session.run(output_names, batch)
    return output

if __name__ == "__main__":
    export_model("models", "model.onnx")
    inference("models/model.onnx")
    visualize_model("models/model.onnx")