"""
run_experiment.py — main entry point for the experiment workflow.

Usage:
    python run_experiment.py --config experiments/configs/02_patchtst_etth1.yaml
    python run_experiment.py --config experiments/configs/06_film_fusion_etth1.yaml \\
        --name ablation_lr --set training.learning_rate=0.001
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from layers.TextEncoder import requires_bert, BERT_MODELS

REGISTRY_PATH = Path('results/iteration3_registry.json')


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply --set key.subkey=value overrides to the config dict."""
    for override in overrides:
        key_path, _, value = override.partition('=')
        keys = key_path.strip().split('.')
        node = config
        for k in keys[:-1]:
            node = node[k]
        leaf = keys[-1]
        # Try to cast to int/float/bool, fallback to string
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        if value in ('true', 'True'):
            value = True
        elif value in ('false', 'False'):
            value = False
        node[leaf] = value
    return config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_gpu_requirement(model_name: str):
    has_gpu = torch.cuda.is_available() or torch.backends.mps.is_available()
    if requires_bert(model_name) and not has_gpu:
        print(
            f'\n  WARNING: {model_name} requires BERT, which is impractical on CPU.\n'
            f'  No GPU found (CUDA or Apple MPS).\n'
            f'  → On Apple Silicon ensure PyTorch ≥ 2.0 is installed.\n',
            file=sys.stderr,
        )
        sys.exit(1)


def make_result_dir(config: dict, name_override: str | None) -> Path:
    exp_name = name_override or config.get('name', 'experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = Path('experiments/results') / f'{exp_name}_{timestamp}'
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def save_config(config: dict, result_dir: Path):
    with open(result_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_metrics(metrics: dict, result_dir: Path):
    with open(result_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


def update_registry(config: dict, metrics: dict, result_dir: Path):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = []

    registry.append({
        'name':           config.get('name', 'experiment'),
        'model':          config['model']['name'],
        'dataset':        config['data']['dataset'],
        'pred_len':       config['model']['pred_len'],
        'seq_len':        config['model']['seq_len'],
        'train_fraction': config['training'].get('train_fraction', 1.0),
        'seed':           config['training'].get('seed', 2024),
        'text_source':    config['model'].get('text_source', None),
        'description':    config.get('description', ''),
        'result_dir':     str(result_dir),
        'timestamp':      result_dir.name.split('_')[-2] + '_' + result_dir.name.split('_')[-1],
        'metrics':        metrics,
    })

    with open(REGISTRY_PATH, 'w') as f:
        json.dump(registry, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(config_path: str, name_override: str | None = None, overrides: list[str] = None):
    config = load_config(config_path)
    if overrides:
        config = apply_overrides(config, overrides)

    model_name = config['model']['name']
    check_gpu_requirement(model_name)

    seed = config['training'].get('seed', 2024)
    set_seed(seed)

    result_dir = make_result_dir(config, name_override)
    save_config(config, result_dir)

    print(f'\nExperiment : {config.get("name", "experiment")}')
    print(f'Model      : {model_name}')
    print(f'Dataset    : {config["data"]["dataset"]}')
    print(f'pred_len   : {config["model"]["pred_len"]}')
    print(f'Results in : {result_dir}\n')

    from exp.exp_forecasting import Exp_Forecasting
    exp = Exp_Forecasting(config)

    # ── Train ──────────────────────────────────────────────────────────────
    exp.train(str(result_dir))

    # ── Test ───────────────────────────────────────────────────────────────
    metrics = exp.test(str(result_dir))
    save_metrics(metrics, result_dir)
    update_registry(config, metrics, result_dir)

    print(f'\nDone. Results saved to {result_dir}')
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run a single experiment from a YAML config.')
    parser.add_argument('--config', required=True,
                        help='Path to YAML config file (e.g. experiments/configs/02_patchtst_etth1.yaml)')
    parser.add_argument('--name', default=None,
                        help='Override experiment name (used for the result directory)')
    parser.add_argument('--set', nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help='Override config values, e.g. --set training.learning_rate=0.001')
    args = parser.parse_args()
    run(args.config, args.name, args.set)


if __name__ == '__main__':
    main()
