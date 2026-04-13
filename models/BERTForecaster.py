import torch
import torch.nn as nn
from layers.TextEncoder import TextEncoder, generate_ts_description


class Model(nn.Module):
    """
    Text-only forecaster.

    Pipeline:
        x_enc  →  generate statistical text description
               →  BERT (frozen)  →  mean-pool  →  [B, 768]
               →  MLP  →  [B, pred_len * enc_in]
               →  reshape  →  [B, pred_len, enc_in]

    Serves as an ablation to measure how much signal the text description alone carries.
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len   = configs.pred_len
        self.enc_in     = configs.enc_in
        self.dataset_name = getattr(configs, 'data_path', 'dataset').split('.')[0]

        text_model  = getattr(configs, 'text_model', 'sentence-transformers/all-MiniLM-L6-v2')
        text_source = getattr(configs, 'text_source', 'template')
        self.text_encoder = TextEncoder(
            model_name=text_model, random_mode=(text_source == 'random'))

        text_hidden = self.text_encoder.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(text_hidden, text_hidden // 2),
            nn.ReLU(),
            nn.Dropout(getattr(configs, 'dropout', 0.1)),
            nn.Linear(text_hidden // 2, self.pred_len * self.enc_in),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, text_emb=None):
        if text_emb is None:
            texts = generate_ts_description(x_enc, self.dataset_name, self.pred_len, x_mark_enc)
            text_emb = self.text_encoder(texts)        # [B, text_hidden]
        out = self.head(text_emb)                      # [B, pred_len * enc_in]
        out = out.view(x_enc.size(0), self.pred_len, self.enc_in)
        return out
