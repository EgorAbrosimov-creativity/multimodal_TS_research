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
    Ensemble Fusion: one TS branch (PatchTST) + one text branch (frozen LM MLP).
    The text embedding predicts a scalar interpolation weight α ∈ (0, 1) per sample:

        ts_out   = PatchTST(x_enc)              [B, pred_len, enc_in]
        text_emb = LM(generate_prompt(x_enc))   [B, text_hidden]
        text_out = MLP(text_emb)                [B, pred_len, enc_in]
        α        = sigmoid(Linear(text_emb))    [B, 1, 1]
        out      = α · ts_out + (1-α) · text_out

    The TS branch uses instance normalization; text_out is predicted in
    normalized space so α learns a meaningful trade-off between the two branches.
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.pred_len = configs.pred_len
        self.enc_in   = configs.enc_in
        self.seq_len  = configs.seq_len
        self.dataset_name = getattr(configs, 'data_path', 'dataset').split('.')[0]

        padding = stride

        # ── TS branch (PatchTST backbone) ────────────────────────────────
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

        # ── Text branch ───────────────────────────────────────────────────
        text_model = getattr(configs, 'text_model', 'bert-base-uncased')
        self.text_encoder = TextEncoder(model_name=text_model)
        text_hidden = self.text_encoder.hidden_dim

        self.text_head = nn.Sequential(
            nn.Linear(text_hidden, text_hidden // 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            nn.Linear(text_hidden // 2, configs.pred_len * configs.enc_in),
        )

        # ── Interpolation weight: text predicts how much to trust text vs TS ─
        self.alpha_proj = nn.Linear(text_hidden, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B = x_enc.size(0)

        # ── Instance normalization ────────────────────────────────────────
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # ── TS branch ────────────────────────────────────────────────────
        x_p = x.permute(0, 2, 1)                              # [B, n_vars, T]
        enc_out, n_vars = self.patch_embedding(x_p)           # [B*n_vars, patch_num, d_model]
        enc_out, _ = self.encoder(enc_out)                    # [B*n_vars, patch_num, d_model]
        patch_num = enc_out.shape[1]
        enc_out = enc_out.reshape(B, n_vars, patch_num, -1)   # [B, n_vars, patch_num, d_model]
        enc_out = enc_out.permute(0, 1, 3, 2)                 # [B, n_vars, d_model, patch_num]
        ts_out = self.ts_head(enc_out).permute(0, 2, 1)       # [B, pred_len, n_vars]

        # ── Text branch (operates on normalized input) ────────────────────
        texts = generate_ts_description(x_enc, self.dataset_name, self.pred_len)
        text_emb = self.text_encoder(texts)                   # [B, text_hidden]
        text_out = self.text_head(text_emb)                   # [B, pred_len * enc_in]
        text_out = text_out.view(B, self.pred_len, self.enc_in)

        # ── Interpolation ─────────────────────────────────────────────────
        alpha = torch.sigmoid(self.alpha_proj(text_emb))      # [B, 1]
        alpha = alpha.unsqueeze(2)                             # [B, 1, 1] → broadcast
        out = alpha * ts_out + (1.0 - alpha) * text_out       # [B, pred_len, enc_in]

        # ── De-normalization ──────────────────────────────────────────────
        out = out * stdev[:, 0, :].unsqueeze(1).expand_as(out)
        out = out + means[:, 0, :].unsqueeze(1).expand_as(out)
        return out
