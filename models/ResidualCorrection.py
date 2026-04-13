import torch
import torch.nn as nn
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
    F10 — Residual Text Correction.

    Structural fix for the LateFusion failure mode: text is prevented from
    overriding the TS signal by design. A PatchTST produces the primary forecast;
    the text branch produces a bounded correction scaled by a learned per-channel
    scalar β, initialized near zero.

    Pipeline:
        ts_out          = PatchTST(x_enc)         [B, pred_len, enc_in]
        text_correction = MLP(text_emb)            [B, pred_len * enc_in]
        output          = ts_out + β * text_correction

    β is a per-channel learnable parameter (shape [enc_in]), initialized to 0.01
    so the model starts as standard PatchTST. Gradient only pushes β away from
    zero when the text correction genuinely reduces the loss.

    Key diagnostic: track β magnitude across training fractions. If β → 0 at all
    fractions, text provides no useful correction. If β grows at low fractions,
    text compensates for reduced TS signal — directly supporting the graceful
    degradation hypothesis.

    self.last_beta is set to β.detach() each forward pass for diagnostics.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in   = configs.enc_in
        self.seq_len  = configs.seq_len
        self.dataset_name = getattr(configs, 'data_path', 'dataset').split('.')[0]

        padding = stride

        # ── PatchTST backbone (primary forecast) ──────────────────────────
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
        self.ts_head = FlattenHead(configs.enc_in, head_nf, configs.pred_len,
                                   head_dropout=configs.dropout)

        # ── Text encoder ──────────────────────────────────────────────────
        text_model  = getattr(configs, 'text_model', 'sentence-transformers/all-MiniLM-L6-v2')
        text_source = getattr(configs, 'text_source', 'template')
        self.text_encoder = TextEncoder(
            model_name=text_model, random_mode=(text_source == 'random'))
        text_hidden = self.text_encoder.hidden_dim

        # ── Text correction MLP ───────────────────────────────────────────
        self.correction_mlp = nn.Sequential(
            nn.Linear(text_hidden, text_hidden // 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(text_hidden // 2, configs.pred_len * configs.enc_in),
        )

        # ── β: per-channel correction scale, initialized near 0 ──────────
        self.beta = nn.Parameter(0.01 * torch.ones(configs.enc_in))

        # Diagnostic
        self.last_beta = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_emb=None):
        B = x_enc.size(0)

        # ── Instance normalization ────────────────────────────────────────
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # ── Primary TS forecast ───────────────────────────────────────────
        x_p = x.permute(0, 2, 1)                               # [B, n_vars, T]
        enc_out, n_vars = self.patch_embedding(x_p)            # [B*n_vars, patch_num, d_model]
        enc_out, _ = self.encoder(enc_out)
        patch_num = enc_out.shape[1]
        enc_out = enc_out.reshape(B, n_vars, patch_num, -1)
        enc_out = enc_out.permute(0, 1, 3, 2)                  # [B, n_vars, d_model, patch_num]
        ts_out = self.ts_head(enc_out).permute(0, 2, 1)        # [B, pred_len, n_vars]

        # ── De-normalize TS output ────────────────────────────────────────
        ts_out = ts_out * stdev[:, 0, :].unsqueeze(1).expand_as(ts_out)
        ts_out = ts_out + means[:, 0, :].unsqueeze(1).expand_as(ts_out)

        # ── Text correction ───────────────────────────────────────────────
        if text_emb is None:
            texts = generate_ts_description(x_enc, self.dataset_name, self.pred_len, x_mark_enc)
            text_emb = self.text_encoder(texts)                # [B, text_hidden]

        correction = self.correction_mlp(text_emb)             # [B, pred_len * enc_in]
        correction = correction.view(B, self.pred_len, self.enc_in)

        # β [enc_in] broadcast over [B, pred_len, enc_in]
        self.last_beta = self.beta.detach()                    # for diagnostics
        out = ts_out + self.beta * correction
        return out
