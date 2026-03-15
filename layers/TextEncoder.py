import numpy as np
import torch
import torch.nn as nn

BERT_MODELS = (
    'BERTForecaster', 'LateFusion', 'GatedFusion',
    'FiLMFusion', 'EnsembleFusion',
    'CrossAttentionFusion', 'ResidualCorrection',
)


def requires_bert(model_name: str) -> bool:
    return model_name in BERT_MODELS


_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
_SEASONS = {
    1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 5: 'spring',
    6: 'summer', 7: 'summer', 8: 'summer', 9: 'autumn', 10: 'autumn',
    11: 'autumn', 12: 'winter',
}


def generate_ts_description(
    x_enc: torch.Tensor,
    dataset_name: str,
    pred_len: int,
    x_mark_enc: torch.Tensor | None = None,
    dataset_stats: dict | None = None,
) -> list[str]:
    """
    Generate enhanced template-based statistical descriptions per batch sample.

    Aggregates statistics across all variables in the window:
      min, max, median, std, slope, regime, autocorr lag-1 & lag-24,
      dominant cycle, last value, recent delta, temporal context.

    Args:
        x_enc:          [B, T, n_vars] — input time series (any device)
        dataset_name:   human-readable label, e.g. "ETTh1"
        pred_len:       forecast horizon
        x_mark_enc:     [B, T, F] time features in [-0.5, 0.5] (optional)
                        F=4 hourly:  [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
                        F=5 15-min:  [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
        dataset_stats:  optional dict with pre-computed percentile thresholds:
                        {'std_p80': float, 'slope_p80': float}

    Returns:
        List of B strings.
    """
    x    = x_enc.detach().cpu()
    B, T, n_vars = x.shape
    mark = x_mark_enc.detach().cpu() if x_mark_enc is not None else None

    std_p80   = dataset_stats.get('std_p80')   if dataset_stats else None
    slope_p80 = dataset_stats.get('slope_p80') if dataset_stats else None

    prompts = []
    for b in range(B):
        window = x[b]                               # [T, n_vars]
        series = window.mean(dim=1).numpy()         # [T]

        # ── Core stats ────────────────────────────────────────────────────
        min_val  = float(window.min())
        max_val  = float(window.max())
        std_val  = float(np.std(series))
        last_val = float(series[-1])
        slope    = (series[-1] - series[0]) / max(T - 1, 1)
        delta    = float(series[-1] - series[-(T // 4) - 1])   # change over last 25%

        # ── Regime ────────────────────────────────────────────────────────
        range_val = max_val - min_val + 1e-8
        if std_p80 is not None and std_val > std_p80:
            regime = 'volatile'
        elif slope_p80 is not None:
            if slope > slope_p80:
                regime = 'rising'
            elif slope < -slope_p80:
                regime = 'falling'
            else:
                regime = 'flat'
        else:
            # fallback thresholds when dataset stats are unavailable
            threshold = 0.005 * range_val / max(T - 1, 1)
            if slope > threshold:
                regime = 'rising'
            elif slope < -threshold:
                regime = 'falling'
            else:
                regime = 'flat'

        # ── Relative language ─────────────────────────────────────────────
        vol_str = ''
        slope_str = ''
        if std_p80 is not None:
            vol_str = f' ({"above" if std_val > std_p80 else "below"} dataset avg)'
        if slope_p80 is not None and regime in ('rising', 'falling'):
            strength = 'strong' if abs(slope) > slope_p80 else 'moderate'
            slope_str = f' ({strength})'

        # ── Autocorrelation ───────────────────────────────────────────────
        if T > 1:
            ac1 = float(np.corrcoef(series[:-1], series[1:])[0, 1])
            if np.isnan(ac1):
                ac1 = 0.0
        else:
            ac1 = 0.0

        ac24_str = ''
        if T > 24:
            ac24 = float(np.corrcoef(series[:-24], series[24:])[0, 1])
            if not np.isnan(ac24):
                ac24_str = f', lag-24={ac24:.2f}'

        # ── Dominant period (FFT) ─────────────────────────────────────────
        centered = series - series.mean()
        fft_mag  = np.abs(np.fft.rfft(centered))
        freqs    = np.fft.rfftfreq(T)
        if len(freqs) > 1 and fft_mag[1:].max() > 0:
            dom_freq   = freqs[1:][np.argmax(fft_mag[1:])]
            dom_period = int(round(1.0 / dom_freq)) if dom_freq > 0 else T
        else:
            dom_period = T

        # ── Temporal context ──────────────────────────────────────────────
        time_str = ''
        if mark is not None:
            m = mark[b, -1].numpy()
            F = len(m)
            try:
                if F == 4:   # hourly: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
                    hour   = int(round((float(m[0]) + 0.5) * 23))
                    dow    = int(round((float(m[1]) + 0.5) * 6))
                    doy    = int(round((float(m[3]) + 0.5) * 365)) + 1
                    month  = max(1, min(12, int(doy / 30.44) + 1))
                    season = _SEASONS.get(month, '')
                    time_str = f' Recorded {_DAYS[dow]} at {hour:02d}:00, {season}.'
                elif F == 5: # 15-min: [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
                    minute = int(round((float(m[0]) + 0.5) * 59))
                    hour   = int(round((float(m[1]) + 0.5) * 23))
                    dow    = int(round((float(m[2]) + 0.5) * 6))
                    doy    = int(round((float(m[4]) + 0.5) * 365)) + 1
                    month  = max(1, min(12, int(doy / 30.44) + 1))
                    season = _SEASONS.get(month, '')
                    time_str = f' Recorded {_DAYS[dow]} at {hour:02d}:{minute:02d}, {season}.'
            except Exception:
                pass  # graceful skip

        # ── Assemble ─────────────────────────────────────────────────────
        prompt = (
            f"Dataset: {dataset_name}. "
            f"Task: forecast the next {pred_len} steps from the past {T} steps. "
            f"{regime.capitalize()} trend{slope_str} (slope={slope:+.4f}). "
            f"Volatility std={std_val:.3f}{vol_str}. "
            f"Dominant cycle: {dom_period} steps. "
            f"Last value {last_val:.3f}, recent change {delta:+.3f}. "
            f"Autocorr lag-1={ac1:.2f}{ac24_str}."
            f"{time_str}"
        )
        prompts.append(prompt)
    return prompts


class TextEncoder(nn.Module):
    """
    Frozen text encoder that converts text descriptions into fixed-size embeddings.

    Supports three backends:
        'bert-base-uncased'                          → 768-dim
        'google/flan-t5-small'                       → 512-dim
        'sentence-transformers/all-MiniLM-L6-v2'    → 384-dim  (default)

    Pass random_mode=True to skip loading the model and return random Gaussian
    embeddings instead (D3 / A1 ablation). The hidden_dim is still set correctly
    from model_name so downstream projection layers are sized properly.

    Usage:
        encoder = TextEncoder()                         # MiniLM-L6 default
        encoder = TextEncoder('bert-base-uncased')      # BERT
        encoder = TextEncoder(random_mode=True)         # random ablation
        emb = encoder(["Dataset: ETTh1. ..."])          # [B, hidden_dim]
    """

    HIDDEN_DIM = {
        'bert-base-uncased': 768,
        'google/flan-t5-small': 512,
        'sentence-transformers/all-MiniLM-L6-v2': 384,
    }

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        random_mode: bool = False,
    ):
        super().__init__()
        if model_name not in self.HIDDEN_DIM:
            raise ValueError(
                f"Unknown text model '{model_name}'. "
                f"Supported: {list(self.HIDDEN_DIM.keys())}"
            )
        self.model_name  = model_name
        self.hidden_dim  = self.HIDDEN_DIM[model_name]
        self.random_mode = random_mode

        if random_mode:
            # Dummy buffer so .device works without loading a real model
            self.register_buffer('_device_ref', torch.zeros(1))
        else:
            from transformers import AutoTokenizer, AutoModel, T5EncoderModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if 'flan-t5' in model_name:
                self._model = T5EncoderModel.from_pretrained(model_name)
            else:
                self._model = AutoModel.from_pretrained(model_name)
            for param in self._model.parameters():
                param.requires_grad = False

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of B strings
        Returns:
            emb: [B, hidden_dim]
                 Random Gaussian when random_mode=True;
                 mean-pooled encoder output otherwise.
        """
        B = len(texts)
        if self.random_mode:
            return torch.randn(B, self.hidden_dim, device=self._device_ref.device)

        device = next(self._model.parameters()).device
        tokens = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128,
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        with torch.no_grad():
            if 'flan-t5' in self.model_name:
                out = self._model(
                    input_ids=tokens['input_ids'],
                    attention_mask=tokens['attention_mask'],
                )
            else:
                out = self._model(**tokens)
            hidden = out.last_hidden_state          # [B, seq, dim]

        mask = tokens['attention_mask'].unsqueeze(-1).float()
        emb  = (hidden * mask).sum(1) / mask.sum(1) # [B, hidden_dim]
        return emb
