import torch
import torch.nn as nn

BERT_MODELS = (
    'BERTForecaster', 'LateFusion', 'GatedFusion',
    'FiLMFusion', 'EnsembleFusion',
)


def requires_bert(model_name: str) -> bool:
    return model_name in BERT_MODELS


def generate_ts_description(x_enc: torch.Tensor, dataset_name: str, pred_len: int) -> list[str]:
    """
    Generate one template-based statistical description per batch sample.

    Aggregates statistics across all variables in the window:
      min, max, median, std, overall trend direction, recent trend direction.

    Args:
        x_enc:        [B, T, n_vars] — input time series (can be on any device)
        dataset_name: human-readable dataset label, e.g. "ETTh1"
        pred_len:     forecast horizon

    Returns:
        List of B strings, one per sample.
    """
    x = x_enc.detach().cpu()
    B, T, n_vars = x.shape
    prompts = []
    for b in range(B):
        window = x[b]  # [T, n_vars]
        min_val   = window.min().item()
        max_val   = window.max().item()
        med_val   = window.median().item()
        std_val   = window.std().item()
        trend     = (window[-1].mean() - window[0].mean()).item()
        direction = "upward" if trend > 0 else "downward"

        # Short-term trend: compare last 25% of window vs the 25-50% block before it
        recent = window[-(T // 4):]
        prev   = window[-(T // 2):-(T // 4)]
        r_dir  = "upward" if recent.mean() > prev.mean() else "downward"

        prompt = (
            f"Dataset: {dataset_name}. "
            f"Task: forecast the next {pred_len} steps from the past {T} steps. "
            f"Stats: min={min_val:.3f}, max={max_val:.3f}, "
            f"median={med_val:.3f}, std={std_val:.3f}. "
            f"Trend: overall {direction}; recent (last {T // 4} steps) {r_dir}."
        )
        prompts.append(prompt)
    return prompts


class TextEncoder(nn.Module):
    """
    Frozen text encoder that converts text descriptions into fixed-size embeddings.

    Supports three backends:
        'bert-base-uncased'                          → 768-dim
        'google/flan-t5-small'                       → 512-dim
        'sentence-transformers/all-MiniLM-L6-v2'    → 384-dim

    Usage:
        encoder = TextEncoder()                          # BERT default
        encoder = TextEncoder('google/flan-t5-small')   # FLAN-T5
        texts = ["Dataset: ETTh1. Task: forecast..."]   # list of B strings
        emb = encoder(texts)   # [B, hidden_dim]
    """

    HIDDEN_DIM = {
        'bert-base-uncased': 768,
        'google/flan-t5-small': 512,
        'sentence-transformers/all-MiniLM-L6-v2': 384,
    }

    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__()
        if model_name not in self.HIDDEN_DIM:
            raise ValueError(
                f"Unknown text model '{model_name}'. "
                f"Supported: {list(self.HIDDEN_DIM.keys())}"
            )
        self.model_name = model_name
        self.hidden_dim = self.HIDDEN_DIM[model_name]

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
            emb: [B, hidden_dim] mean-pooled over non-padding token positions
        """
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
                hidden = out.last_hidden_state          # [B, seq, 512]
            else:
                out = self._model(**tokens)
                hidden = out.last_hidden_state          # [B, seq, dim]

        # mean-pool over non-padding tokens
        mask = tokens['attention_mask'].unsqueeze(-1).float()
        emb = (hidden * mask).sum(1) / mask.sum(1)     # [B, hidden_dim]
        return emb
