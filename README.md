# Multimodal Time-Series Forecasting

Research codebase comparing TS-only baselines against text-augmented fusion models on time-series forecasting benchmarks.

---

## Overview

Text descriptions are generated **automatically** from each input window's statistics (min, max, median, trend direction). No external text data is required. BERT encodes these descriptions; the resulting embeddings are fused with time-series features in different ways to test whether textual summaries help forecasting.

---

## Model Lineup

| # | Model | Type | GPU needed |
|---|-------|------|-----------|
| 1 | **DLinear** | TS-only baseline | No (CPU ok) |
| 2 | **PatchTST** ⭐ | TS-only backbone | Recommended |
| 3 | **BERTForecaster** | Text-only (ablation) | Yes — Colab |
| 4 | **LateFusion** | Concat TS + text outputs → MLP | Yes — Colab |
| 5 | **GatedFusion** ⭐ | Text gates TS encoder features | Yes — Colab |
| 6 | **FiLMFusion** ⭐ | Text generates γ, β for TS features | Yes — Colab |

---

## Project Structure

```
multimodality/
├── run_experiment.py          # main entry point
├── compare_results.py         # compare results across experiments
├── requirements.txt
│
├── models/
│   ├── DLinear.py             # existing baseline
│   ├── PatchTST.py            # existing backbone
│   ├── BERTForecaster.py      # text-only model
│   ├── LateFusion.py          # output-level fusion
│   ├── GatedFusion.py         # feature-level gating
│   └── FiLMFusion.py          # feature-level FiLM modulation
│
├── layers/
│   ├── TextEncoder.py         # text generation + frozen BERT wrapper
│   └── ...                    # existing layers
│
├── exp/
│   ├── exp_basic.py           # base experiment class
│   └── exp_forecasting.py     # training / validation / test loop
│
├── data_provider/
│   ├── data_loader.py         # Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
│   └── data_factory.py        # data_provider(config, flag) factory
│
├── dataset/
│   ├── ETT-small/             # ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
│   ├── electricity/
│   ├── weather/
│   └── ...
│
└── experiments/
    ├── registry.json          # auto-updated index of all runs
    ├── configs/               # one YAML file per experiment
    │   ├── 01_dlinear_etth1.yaml
    │   ├── 02_patchtst_etth1.yaml
    │   ├── 03_bert_forecaster_etth1.yaml
    │   ├── 04_late_fusion_etth1.yaml
    │   ├── 05_gated_fusion_etth1.yaml
    │   └── 06_film_fusion_etth1.yaml
    └── results/               # auto-created, one folder per run
        └── {name}_{timestamp}/
            ├── config.yaml    # exact config used
            ├── metrics.json   # mae, mse, rmse, mape, mspe
            ├── checkpoint.pth # best model weights
            ├── preds.npy
            └── trues.npy
```

---

## Datasets

Datasets are not included in this repo. Download them from the [TimeVLM repository](https://github.com/CityMind-Lab/ICML25-TimeVLM/tree/main/src/TimeVLM) and place them under `dataset/`:

```
dataset/
├── ETT-small/      # ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
├── electricity/
├── traffic/
├── weather/
├── illness/
└── exchange_rate/
```

---

## Setup

```bash
pip install -r requirements.txt
```

BERT-based models (configs 03–06) require a GPU. Connect a Colab GPU runtime via the VSCode extension, then run commands as normal.

---

## Running Experiments

```bash
# Run a single experiment
python run_experiment.py --config experiments/configs/02_patchtst_etth1.yaml

# Override config values without editing the YAML
python run_experiment.py --config experiments/configs/02_patchtst_etth1.yaml \
    --name patchtst_lr_sweep \
    --set training.learning_rate=0.001 training.train_epochs=10

# Compare all completed experiments
python compare_results.py

# Filter and sort
python compare_results.py --filter model=PatchTST
python compare_results.py --filter dataset=ETTh1 --sort mse
```

---

## Adding a New Experiment

1. Copy an existing YAML from `experiments/configs/` and edit it.
2. Set `name`, `model.name`, `data.*`, and `training.*` as needed.
3. Run `python run_experiment.py --config experiments/configs/your_config.yaml`.

Results land in `experiments/results/{name}_{timestamp}/` and are added to `experiments/registry.json` automatically.

---

## Adding a New Model

1. Create `models/YourModel.py` with a `Model(configs)` class.
   - `configs` is a `SimpleNamespace` with all YAML keys flattened (e.g. `configs.seq_len`, `configs.d_model`).
   - Implement `forward(x_enc, x_mark_enc, x_dec, x_mark_dec)` → `[B, pred_len, enc_in]`.
2. If your model uses BERT, import `TextEncoder` and `generate_ts_description` from `layers/TextEncoder.py`, and add your model name to `BERT_MODELS` in that file.
3. Create a YAML config referencing `model.name: YourModel`.

---

## Text Generation

Statistical descriptions are generated per sample inside each fusion model's `forward()`:

```
"Dataset: ETTh1. Task: forecast the next 96 steps from the past 336 steps.
 Input statistics: min=-1.243, max=2.107, median=0.412, overall trend is upward."
```

This happens automatically — the data pipeline is unchanged for all models.

---

## Architecture Details

### GatedFusion
Text embedding generates a soft gate `σ(W · text_emb)` ∈ [0,1]^{d_model}, applied element-wise to PatchTST encoder features before the prediction head. The gate learns which feature dimensions the text description is most informative about.

### FiLMFusion
Text embedding predicts affine parameters (γ, β) via two linear layers. These modulate PatchTST encoder features as `γ ⊙ enc_out + β` (Feature-wise Linear Modulation). Unlike gating, FiLM can both scale and shift features.

---

## Dataset Splits

ETT datasets use the standard literature splits (same as TimeVLM / PatchTST papers):

| Dataset | Train | Val | Test |
|---------|-------|-----|------|
| ETTh1/h2 | 12 months | 4 months | 4 months |
| ETTm1/m2 | 12 months | 4 months | 4 months |

Custom datasets use a 70 / 10 / 20 split.
