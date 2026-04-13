"""
generate_configs.py — combinatorial YAML config generator for Iteration 2.

Generates experiment config files for:
  --track d_series   D1/D2/D3 description ablation (72 configs)
  --track tier1      Full fraction sweep, Tier 1 datasets (~270 config files)
  --track tier2      Extended datasets, 100% + 10% only

Usage:
    python generate_configs.py --track d_series
    python generate_configs.py --track tier1
    python generate_configs.py --track tier1 --models PatchTST EnsembleFusion --datasets ETTh1
    python generate_configs.py --track tier2

Output: experiments/configs/{prefix}_{model}_{dataset}_f{fraction}_h{horizon}_s{seed}.yaml

Naming:
    d_series: d_{model}_{dataset}_src{source}_f{fraction}_h{horizon}_s{seed}.yaml
    tier1:    t1_{model}_{dataset}_f{fraction}_h{horizon}_s{seed}.yaml
    tier2:    t2_{model}_{dataset}_f{fraction}_h{horizon}_s{seed}.yaml
"""

import argparse
import os
from pathlib import Path

import yaml

OUT_DIR = Path('experiments/configs')

# ── Dataset definitions ───────────────────────────────────────────────────────

DATASET_CFG = {
    'ETTh1': {
        'dataset': 'ETTh1',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTh1.csv',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
        'enc_in': 7,
    },
    'ETTh2': {
        'dataset': 'ETTh2',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTh2.csv',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
        'enc_in': 7,
    },
    'ETTm1': {
        'dataset': 'ETTm1',
        'root_path': './dataset/ETT-small/',
        'data_path': 'ETTm1.csv',
        'freq': 't',
        'features': 'M',
        'target': 'OT',
        'enc_in': 7,
    },
    'Weather': {
        'dataset': 'custom',
        'root_path': './dataset/weather/',
        'data_path': 'weather.csv',
        'freq': 'h',
        'features': 'M',
        'target': 'OT',
        'enc_in': 21,
    },
    'ExchangeRate': {
        'dataset': 'custom',
        'root_path': './dataset/exchange_rate/',
        'data_path': 'exchange_rate.csv',
        'freq': 'd',
        'features': 'M',
        'target': 'OT',
        'enc_in': 8,
    },
}

# ── Model defaults ────────────────────────────────────────────────────────────

# Models that use text — must include diagnostics fields and text config
TEXT_MODELS = {
    'GatedFusion', 'FiLMFusion', 'EnsembleFusion',
    'BERTForecaster', 'LateFusion',
    'CrossAttentionFusion', 'ResidualCorrection',
}

# Seeds: 3 for primary, 1 for deprecated/ablation-only
SEEDS = {
    'DLinear': [2024, 2025, 2026],
    'PatchTST': [2024, 2025, 2026],
    'GatedFusion': [2024, 2025, 2026],
    'FiLMFusion': [2024, 2025, 2026],
    'EnsembleFusion': [2024, 2025, 2026],
    'CrossAttentionFusion': [2024, 2025, 2026],
    'ResidualCorrection': [2024, 2025, 2026],
    'BERTForecaster': [2024],
    'LateFusion': [2024],
}

# Diagnostics flags per model
DIAG_FLAGS = {
    'EnsembleFusion': {'log_alpha': True},
    'GatedFusion': {'log_gates': True},
    'ResidualCorrection': {'log_beta': True},
    'CrossAttentionFusion': {'log_attn': True},
}

TEXT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TEXT_HIDDEN = 384


def model_cfg(model_name: str, enc_in: int, seq_len: int,
              pred_len: int, text_source: str = 'template') -> dict:
    cfg = {
        'name': model_name,
        'task_name': 'long_term_forecast',
        'enc_in': enc_in,
        'seq_len': seq_len,
        'label_len': 48,
        'pred_len': pred_len,
        'd_model': 128,
        'n_heads': 16,
        'e_layers': 3,
        'd_layers': 1,
        'd_ff': 768,
        'factor': 1,
        'dropout': 0.2,
        'activation': 'gelu',
        'moving_avg': 25,
    }
    if model_name in TEXT_MODELS:
        cfg['text_model'] = TEXT_MODEL_NAME
        cfg['text_hidden'] = TEXT_HIDDEN
        cfg['text_source'] = text_source
    return cfg


def training_cfg(seed: int, fraction: float,
                 batch_size: int = 32, lr: float = 0.0001,
                 epochs: int = 15, patience: int = 5) -> dict:
    cfg = {
        'train_epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': lr,
        'patience': patience,
        'seed': seed,
    }
    if fraction < 1.0:
        cfg['train_fraction'] = fraction
    return cfg


def compute_cfg() -> dict:
    return {'gpu': 0, 'num_workers': 4, 'use_amp': True}


def diag_cfg(model_name: str):
    flags = DIAG_FLAGS.get(model_name)
    if not flags:
        return None
    return {'enabled': True, **flags}


def fraction_str(f: float) -> str:
    """Format fraction as string for filenames, e.g. 1.0 → '100', 0.05 → '05'."""
    pct = int(round(f * 100))
    return str(pct)


def write_config(path: Path, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ── Track generators ──────────────────────────────────────────────────────────

def _emb_path(emb_dir: str, ds_name: str) -> str:
    """Build text_emb_path for the train split. Factory derives val/test from this."""
    return f'{emb_dir.rstrip("/")}/{ds_name}_train_minilm.npy'


def gen_d_series(models: list[str], datasets: list[str],
                 fractions: list[float], horizons: list[int],
                 text_sources: list[str], emb_dir: str = None) -> int:
    """D-series: description quality ablation."""
    count = 0
    for model_name in models:
        seeds = SEEDS.get(model_name, [2024, 2025, 2026])
        for ds_name in datasets:
            ds = DATASET_CFG[ds_name]
            for src in text_sources:
                for frac in fractions:
                    for horizon in horizons:
                        for seed in seeds:
                            exp_name = (
                                f"d_{model_name.lower()}_{ds_name.lower()}"
                                f"_src{src}_f{fraction_str(frac)}"
                                f"_h{horizon}_s{seed}"
                            )
                            data_cfg = {k: v for k, v in ds.items() if k != 'enc_in'}
                            if emb_dir and src != 'random' and model_name in TEXT_MODELS:
                                data_cfg['text_emb_path'] = _emb_path(emb_dir, ds_name)
                            config = {
                                'name': exp_name,
                                'description': (
                                    f"D-series ablation: {model_name} on {ds_name} "
                                    f"text_source={src} fraction={frac} pred={horizon} seed={seed}"
                                ),
                                'model': model_cfg(model_name, ds['enc_in'], 512, horizon, src),
                                'data': data_cfg,
                                'training': training_cfg(seed, frac, batch_size=32),
                                'compute': compute_cfg(),
                            }
                            diag = diag_cfg(model_name)
                            if diag:
                                config['diagnostics'] = diag
                            fname = OUT_DIR / f'{exp_name}.yaml'
                            write_config(fname, config)
                            count += 1
    return count


def gen_fraction_sweep(prefix: str, models: list[str], datasets: list[str],
                       fractions: list[float], horizons: list[int],
                       text_source: str = 'template', emb_dir: str = None) -> int:
    """Generic fraction sweep generator (used for Tier 1 and Tier 2)."""
    count = 0
    for model_name in models:
        seeds = SEEDS.get(model_name, [2024, 2025, 2026])
        for ds_name in datasets:
            ds = DATASET_CFG[ds_name]
            for frac in fractions:
                for horizon in horizons:
                    for seed in seeds:
                        exp_name = (
                            f"{prefix}_{model_name.lower()}_{ds_name.lower()}"
                            f"_f{fraction_str(frac)}_h{horizon}_s{seed}"
                        )
                        data_cfg = {k: v for k, v in ds.items() if k != 'enc_in'}
                        if emb_dir and text_source != 'random' and model_name in TEXT_MODELS:
                            data_cfg['text_emb_path'] = _emb_path(emb_dir, ds_name)
                        config = {
                            'name': exp_name,
                            'description': (
                                f"{prefix}: {model_name} on {ds_name} "
                                f"fraction={frac} pred={horizon} seed={seed}"
                            ),
                            'model': model_cfg(model_name, ds['enc_in'], 512, horizon, text_source),
                            'data': data_cfg,
                            'training': training_cfg(seed, frac, batch_size=32),
                            'compute': compute_cfg(),
                        }
                        diag = diag_cfg(model_name)
                        if diag:
                            config['diagnostics'] = diag
                        fname = OUT_DIR / f'{exp_name}.yaml'
                        write_config(fname, config)
                        count += 1
    return count


# ── Main ──────────────────────────────────────────────────────────────────────

ALL_MODELS_TIER1 = [
    'DLinear', 'PatchTST',
    'BERTForecaster', 'LateFusion',
    'GatedFusion', 'FiLMFusion', 'EnsembleFusion',
    'CrossAttentionFusion', 'ResidualCorrection',
]

D_SERIES_MODELS   = ['GatedFusion', 'CrossAttentionFusion']
D_SERIES_SOURCES  = ['template', 'llm', 'random']
D_SERIES_DATASETS = ['ETTh1']
D_SERIES_FRACS    = [1.0, 0.1]
D_SERIES_HORIZONS = [96, 336]

TIER1_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1']
TIER1_FRACS    = [1.0, 0.5, 0.25, 0.1, 0.05]
TIER1_HORIZONS = [96, 336]

TIER2_DATASETS = ['Weather', 'ExchangeRate']
TIER2_FRACS    = [1.0, 0.1]
TIER2_HORIZONS = [96, 336]


def main():
    parser = argparse.ArgumentParser(description='Generate Iteration 2 YAML configs.')
    parser.add_argument('--track', required=True,
                        choices=['d_series', 'tier1', 'tier2'],
                        help='Which experiment track to generate')
    parser.add_argument('--models', nargs='*', default=None,
                        help='Restrict to specific models (default: all for track)')
    parser.add_argument('--datasets', nargs='*', default=None,
                        help='Restrict to specific datasets (default: all for track)')
    parser.add_argument('--text_source', default='template',
                        choices=['template', 'llm', 'random'],
                        help='Text source for tier1/tier2 (default: template)')
    parser.add_argument('--emb_dir', default=None,
                        help='Directory containing pre-encoded .npy embeddings. '
                             'If set, text_emb_path is added to all text-model configs. '
                             'Example: /content/drive/MyDrive/multimodal_TS_research/iteration_2/embeddings')
    args = parser.parse_args()

    if args.track == 'd_series':
        models   = args.models   or D_SERIES_MODELS
        datasets = args.datasets or D_SERIES_DATASETS
        n = gen_d_series(models, datasets, D_SERIES_FRACS, D_SERIES_HORIZONS, D_SERIES_SOURCES,
                         emb_dir=args.emb_dir)
        print(f'D-series: wrote {n} configs to {OUT_DIR}/')

    elif args.track == 'tier1':
        models   = args.models   or ALL_MODELS_TIER1
        datasets = args.datasets or TIER1_DATASETS
        n = gen_fraction_sweep('t1', models, datasets, TIER1_FRACS, TIER1_HORIZONS, args.text_source,
                               emb_dir=args.emb_dir)
        print(f'Tier 1: wrote {n} configs to {OUT_DIR}/')

    elif args.track == 'tier2':
        models   = args.models   or ALL_MODELS_TIER1
        datasets = args.datasets or TIER2_DATASETS
        n = gen_fraction_sweep('t2', models, datasets, TIER2_FRACS, TIER2_HORIZONS, args.text_source,
                               emb_dir=args.emb_dir)
        print(f'Tier 2: wrote {n} configs to {OUT_DIR}/')


if __name__ == '__main__':
    main()
