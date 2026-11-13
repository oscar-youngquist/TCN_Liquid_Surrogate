"""Load a frozen portable state-dict artifact with the 12-value streaming interface."""
import torch
from .checkpoint import load_checkpoint
from .model import build_model, ModelConfig, CachedInference
from .normalization import Normalization


def load_export(path, environments=1, device='cpu'):
    saved = load_checkpoint(path, device)
    model = build_model(ModelConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model'])
    model.requires_grad_(False).eval()
    return CachedInference(model, Normalization(saved['normalization']), environments)
