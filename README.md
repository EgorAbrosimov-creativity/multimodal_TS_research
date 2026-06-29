# Text-Augmented Time Series Forecasting

Research codebase for the study: **do text-augmented fusion models degrade more gracefully than TS-only baselines as training data shrinks, and does the choice of fusion mechanism determine that robustness?**

Grounding paper: Time-VLM (Zhong et al., arXiv:2502.04395, ICML 2025). Scope here is narrower — text + time series only, no vision.

---

## Key Findings

Across 700+ experiments (5 datasets, 5 training fractions, 2 horizons, 7 architectures):

- Fusion models do **not** uniformly outperform PatchTST under data scarcity.
- The two most graceful fusion models (EnsembleFusion, GatedFusion) achieve that graceful degradation by **suppressing the text signal entirely** — not by exploiting it.
- FiLMFusion is the only standard fusion model that demonstrably uses text (∼7% MSE gap over random embeddings), yet it is the **least robust** at short horizons — affine modulation parameters overfit rapidly under scarcity.
- CrossAttentionFusion and ResidualCorrection (fixed architectures) show genuine graceful degradation (ratio 1.10–1.29) closely tracking PatchTST.
- Template-generated descriptions match LLM-generated text (Phi-4-mini, 4-bit) in forecasting accuracy at zero inference cost.
- DLinear collapses catastrophically at 5% training data (ratio ≈ 6×); PatchTST is the correct TS-only comparator for fusion models.

**Revised framing:** the appropriate question is not which fusion models degrade least, but which fusion mechanisms can *exploit* text under data scarcity — currently none do so reliably.

---

## Model Stack

| ID | Model | Type | Text? | Status |
|----|-------|------|-------|--------|
| 01 | DLinear | TS-only baseline | No | Active |
| 02 | PatchTST | TS-only backbone | No | Active |
| 03 | BERTForecaster | Text-only ablation | Yes | Non-viable standalone |
| 05 | GatedFusion | Feature-level gating | Yes | Active — suppresses text (gate near-uniform) |
| 06 | FiLMFusion | Feature-level FiLM modulation | Yes | Active — best semantic sensitivity; overfits at low data |
| 07 | EnsembleFusion | Output-level weighted avg | Yes | Active — graceful via α-collapse, not text use |
| F8 | CrossAttentionFusion | Patch-level cross-attention | Yes | Fixed (N=4 token projection) — validated |
| F10 | ResidualCorrection | Residual text correction | Yes | Fixed (β init 0.1) — validated |

LateFusion (04) deprecated — structurally unstable.

---

## Experiment Summary

| Iteration | Runs | Scope | Key outcome |
|-----------|------|-------|-------------|
| 1 | ~50 | ETTh1, 100% data, models 01–07 | Baseline fusion comparison; LateFusion deprecated |
| 2 | 762 | All Tier 1 datasets (ETTh1/h2/m1), all fractions, seeds | Data-scarcity sweep; α-collapse and gate suppression identified |
| 3 | 468 | D1 text ablation + Tier 2 (Weather, ExchangeRate) + F8/F10 validation | Template wins; cross-dataset consistency confirmed; F8/F10 fixed |

---

## Evaluation Protocol

- **Tier 1 datasets:** ETTh1, ETTh2, ETTm1
- **Tier 2 datasets:** Weather, ExchangeRate
- **Horizons:** `pred_len ∈ {96, 336}`
- **Training fractions:** 100% → 50% → 25% → 10% → 5% (most recent X% of train split)
- **Seeds:** 3 per experiment (2024, 2025, 2026); 1 for BERTForecaster
- **Metrics:** MSE + MAE (mean ± std across seeds)
- **Text source:** `template` (locked after Iter 3 ablation)

---

## Text Encoding Pipeline

Text descriptions are generated offline per window, encoded once with `microsoft/MiniLM-L6-H384-uncased` (22M params, dim 384), and saved as `.npy` files. No LLM inference at training time.

Description template:
```
[REGIME] trend (slope=[X]). Volatility std=[X]. Dominant cycle: [X] steps.
Last value [X], recent change [X]. Autocorr lag-1=[X], lag-24=[X].
Recorded [DAY] at [HOUR]:00.
```

Pre-encoded embeddings live in `embeddings/{source}/{dataset}_{split}_minilm.npy`.

---

## Project Structure

```
models/          — model implementations (DLinear, PatchTST, fusion models)
layers/          — reusable layers (TextEncoder, CrossAttention, etc.)
exp/             — experiment runners (exp_basic.py, exp_forecasting.py)
utils/
  data_provider/ — data loading and factory
  text/          — encode_descriptions (offline embedding pre-computation)
  experiment/    — generate_configs (YAML config grid generator)
  eval/          — metrics
  training/      — losses, EarlyStopping, LR schedule
embeddings/      — pre-encoded MiniLM embeddings (template/, llm/)
dataset/         — ETT-small, weather, exchange_rate, electricity, traffic
results/         — iteration registries (JSON + parquet)
figures/         — analysis figures
run_experiment.py          — main entry point
iteration3_analysis.ipynb  — full results analysis (all 3 iterations)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Tested on MacBook Pro M5 Pro (24GB, MPS backend) and Google Colab T4.

---

## Datasets

Not included. Download from the [Time-VLM repository](https://github.com/CityMind-Lab/ICML25-TimeVLM/tree/main/src/TimeVLM) and place under `dataset/`:

```
dataset/
├── ETT-small/      # ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
├── weather/
├── exchange_rate/
├── electricity/
├── traffic/
└── illness/
```

---

## Running an Experiment

```bash
# Single run
python run_experiment.py --config path/to/config.yaml

# Generate a config grid (e.g. Iter 3 d-series)
python utils/experiment/generate_configs.py --track d_series --models filmfusion gatedfusion

# Compare results
python compare_results.py --filter model=FiLMFusion --sort mse
```

---

## Data Truncation Convention

Training fractions take the **most recent X%** of the train split (adjacent to the val boundary):

```python
start_idx = int(n_train * (1 - fraction))
train_subset = train_data[start_idx:]   # correct
# train_data[:int(n_train * fraction)]  # wrong — maximises temporal gap to test
```
