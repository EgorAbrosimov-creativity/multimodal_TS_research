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
    Gated Fusion: text embedding generates a soft gate over PatchTST encoder features.

    Pipeline:
        enc_out  = PatchTST_Encoder(x_enc)     [B, n_vars, d_model, patch_num]
        text_emb = BERT(x_enc)                  [B, 768]
        gate     = σ(Linear(text_emb))          [B, d_model] → broadcast [B, 1, d_model, 1]
        enc_out  = gate * enc_out               element-wise gate
        out      = PatchTST_Head(enc_out)        [B, pred_len, n_vars]
    """

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

        # ── Text encoder ─────────────────────────────────────────────────
        text_model = getattr(configs, 'text_model', 'bert-base-uncased')
        self.text_encoder = TextEncoder(model_name=text_model)
        text_hidden = self.text_encoder.hidden_dim

        # Gate: text_emb → d_model gate values
        self.gate_proj = nn.Linear(text_hidden, configs.d_model)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B = x_enc.size(0)

        # ── Normalization ─────────────────────────────────────────────────
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # ── PatchTST encoder ──────────────────────────────────────────────
        x = x.permute(0, 2, 1)                               # [B, n_vars, T]
        enc_out, n_vars = self.patch_embedding(x)             # [B*n_vars, patch_num, d_model]
        enc_out, _ = self.encoder(enc_out)                    # [B*n_vars, patch_num, d_model]

        # Reshape for gating: [B, n_vars, patch_num, d_model]
        patch_num = enc_out.shape[1]
        enc_out = enc_out.reshape(B, n_vars, patch_num, -1)

        # ── Gating ────────────────────────────────────────────────────────
        texts = generate_ts_description(x_enc, self.dataset_name, self.pred_len)
        text_emb = self.text_encoder(texts)                   # [B, 768]
        gate = torch.sigmoid(self.gate_proj(text_emb))        # [B, d_model]
        gate = gate.unsqueeze(1).unsqueeze(2)                 # [B, 1, 1, d_model]
        enc_out = gate * enc_out                              # [B, n_vars, patch_num, d_model]

        # ── Prediction head ───────────────────────────────────────────────
        enc_out = enc_out.permute(0, 1, 3, 2)                # [B, n_vars, d_model, patch_num]
        out = self.head(enc_out).permute(0, 2, 1)            # [B, pred_len, n_vars]

        # ── De-normalization ──────────────────────────────────────────────
        out = out * stdev[:, 0, :].unsqueeze(1).expand_as(out)
        out = out + means[:, 0, :].unsqueeze(1).expand_as(out)
        return out
