import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.TextEncoder import TextEncoder, generate_ts_description
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # [B, n_vars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    F8 — Cross-Attention Fusion.

    Each TS patch independently attends to a small set of text tokens derived from
    the text embedding. Different patches can weight text differently, enabling
    patch-specific text conditioning — unlike gate/FiLM/ensemble which apply a
    uniform text influence.

    Pipeline:
        ts_patches  = PatchTST_encoder(x_enc)       [B*n_vars, patch_num, d_model]
        text_tokens = Linear(text_hidden, N*d_model) [B, N, d_model]  (N=4 tokens)
        attn_out    = CrossAttention(Q=ts_patches, K=text_tokens, V=text_tokens)
        ts_aug      = ts_patches + attn_out          residual — text cannot override TS
        out         = FlattenHead(ts_aug)            [B, pred_len, n_vars]

    Key design choices:
    - N=4 text tokens avoids degenerate single-token attention (all patches see
      identical key/value → reduces to learned scalar weighting).
    - Residual addition ensures the model starts as PatchTST and can learn to
      ignore text if it provides no signal.
    - Attention weights [B*n_vars, n_heads, patch_num, N] are stored as
      self.last_attn_weights for diagnostics.
    """

    N_TEXT_TOKENS = 4   # number of text tokens projected from text_emb

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in   = configs.enc_in
        self.seq_len  = configs.seq_len
        self.dataset_name = getattr(configs, 'data_path', 'dataset').split('.')[0]

        padding = stride

        # ── PatchTST backbone ─────────────────────────────────────────────
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )
        head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        self.head = FlattenHead(configs.enc_in, head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

        # ── Text encoder ──────────────────────────────────────────────────
        text_model  = getattr(configs, 'text_model', 'sentence-transformers/all-MiniLM-L6-v2')
        text_source = getattr(configs, 'text_source', 'template')
        self.text_encoder = TextEncoder(
            model_name=text_model, random_mode=(text_source == 'random'))
        text_hidden = self.text_encoder.hidden_dim

        # ── Text → N tokens projection ────────────────────────────────────
        self.text_to_tokens = nn.Linear(text_hidden, self.N_TEXT_TOKENS * configs.d_model)

        # ── Cross-attention (patch queries, text keys/values) ─────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=configs.d_model,
            num_heads=configs.n_heads,
            dropout=configs.dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(configs.d_model)

        # Diagnostic storage
        self.last_attn_weights = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_emb=None):
        B = x_enc.size(0)

        # ── Normalization ─────────────────────────────────────────────────
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # ── PatchTST encoder ──────────────────────────────────────────────
        x = x.permute(0, 2, 1)                                     # [B, n_vars, T]
        enc_out, n_vars = self.patch_embedding(x)                   # [B*n_vars, patch_num, d_model]
        enc_out, _ = self.encoder(enc_out)                          # [B*n_vars, patch_num, d_model]
        patch_num = enc_out.shape[1]

        # ── Text embedding → N tokens ──────────────────────────────────────
        if text_emb is None:
            texts = generate_ts_description(x_enc, self.dataset_name, self.pred_len, x_mark_enc)
            text_emb = self.text_encoder(texts)                     # [B, text_hidden]

        # Project to N text tokens: [B, N, d_model]
        text_tokens = self.text_to_tokens(text_emb)                 # [B, N*d_model]
        text_tokens = text_tokens.view(B, self.N_TEXT_TOKENS, -1)  # [B, N, d_model]

        # Expand text tokens to match batch dimension of enc_out: [B*n_vars, N, d_model]
        text_tokens_exp = text_tokens.unsqueeze(1).expand(
            B, n_vars, self.N_TEXT_TOKENS, enc_out.shape[-1]
        ).reshape(B * n_vars, self.N_TEXT_TOKENS, enc_out.shape[-1])

        # ── Cross-attention: each patch queries text tokens ────────────────
        attn_out, attn_weights = self.cross_attn(
            query=enc_out,          # [B*n_vars, patch_num, d_model]
            key=text_tokens_exp,    # [B*n_vars, N, d_model]
            value=text_tokens_exp,  # [B*n_vars, N, d_model]
        )
        # attn_weights: [B*n_vars, patch_num, N]

        # Store mean attention for diagnostics (averaged over batch and n_vars)
        self.last_attn_weights = attn_weights.detach().reshape(
            B, n_vars, patch_num, self.N_TEXT_TOKENS).mean(dim=(0, 1))  # [patch_num, N]

        # Residual: text can only add to, not override, TS features
        enc_out = self.cross_attn_norm(enc_out + attn_out)          # [B*n_vars, patch_num, d_model]

        # ── Prediction head ───────────────────────────────────────────────
        enc_out = enc_out.reshape(B, n_vars, patch_num, -1)         # [B, n_vars, patch_num, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)                       # [B, n_vars, d_model, patch_num]
        out = self.head(enc_out).permute(0, 2, 1)                   # [B, pred_len, n_vars]

        # ── De-normalization ──────────────────────────────────────────────
        out = out * stdev[:, 0, :].unsqueeze(1).expand_as(out)
        out = out + means[:, 0, :].unsqueeze(1).expand_as(out)
        return out
