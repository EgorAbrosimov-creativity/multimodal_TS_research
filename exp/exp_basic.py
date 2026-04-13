import importlib
import types
import torch
import torch.nn as nn


def config_to_namespace(config: dict) -> types.SimpleNamespace:
    """
    Flatten a nested config dict into a SimpleNamespace so models can access
    attributes via configs.seq_len, configs.d_model, etc.
    """
    ns = types.SimpleNamespace()
    for section in config.values():
        if isinstance(section, dict):
            for k, v in section.items():
                setattr(ns, k, v)
        # top-level scalars (shouldn't normally appear, but handle gracefully)
    # also expose the name directly
    if 'model' in config:
        setattr(ns, 'name', config['model'].get('name', ''))
    return ns


class Exp_Basic:
    def __init__(self, config: dict):
        self.config = config
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self) -> torch.device:
        gpu_id = self.config.get('compute', {}).get('gpu', 0)
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            print(f'Using GPU: cuda:{gpu_id}')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            print('Using GPU: Apple MPS')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device

    def _build_model(self) -> nn.Module:
        model_name = self.config['model']['name']
        module = importlib.import_module(f'models.{model_name}')
        cfg_ns = config_to_namespace(self.config)
        model = module.Model(cfg_ns)
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Model: {model_name} | '
              f'Total params: {n_params:,} | Trainable: {n_trainable:,}')
        return model

    def train(self, result_dir: str) -> dict:
        raise NotImplementedError

    def test(self, result_dir: str) -> dict:
        raise NotImplementedError
