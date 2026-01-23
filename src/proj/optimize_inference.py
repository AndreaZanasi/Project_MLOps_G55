import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import hydra
from omegaconf import DictConfig
from pathlib import Path
import logging

from model import Model

log = logging.getLogger(__name__)

def load_model(cfg: DictConfig, model_path: Path) -> nn.Module:
    """Instantiates the model and loads weights from the provided path."""
    log.info(f"Loading model architecture...")
    model = Model(cfg)
    
    log.info(f"Loading state dict from {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def prune_model(model: nn.Module, amount: float = 0.3):
    """Applies L1 Unstructured pruning to Conv2d and Linear layers."""
    log.info(f"Pruning model with amount={amount}...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent to remove re-parametrization for ONNX export
            prune.remove(module, 'weight')

def quantize_model_dynamic(model: nn.Module) -> nn.Module:
    """Applies dynamic quantization (Linear layers only) for CPU inference."""
    log.info("Applying dynamic quantization...")
    # Dynamic quantization is for Linear/RNNs on CPU
    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def export_to_onnx(model: nn.Module, output_path: str, input_shape: tuple):
    """Exports the PyTorch model to ONNX format."""
    log.info(f"Exporting to ONNX at {output_path}...")
    
    # Shape: (Batch_Size, Channels, Height, Width) -> (1, 1, 64, 1168)
    dummy_input = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    log.info("ONNX export complete.")

@hydra.main(config_path="../../configs", config_name="hydra_cfg.yaml", version_base="1.1")
def main(cfg: DictConfig):
    
    model_dir = Path(cfg.paths.model_dir)
    model_name = cfg.paths.model_name
    model_path = model_dir / model_name
    output_onnx = model_dir / model_name.replace(".pth", ".onnx")

    model = load_model(cfg, model_path)

    if cfg.get("prune", False):
        prune_model(model, amount=0.3)

    if cfg.get("quantize", False):
        model = quantize_model_dynamic(model)

    export_to_onnx(model, str(output_onnx), (1, 64, 1168))

if __name__ == "__main__":
    main()
