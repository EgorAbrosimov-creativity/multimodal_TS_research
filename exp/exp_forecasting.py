import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from exp.exp_basic import Exp_Basic
from utils.data_provider.data_factory import data_provider
from utils.eval.metrics import metric


class EarlyStopping:
    def __init__(self, patience: int, checkpoint_path: str):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = float('inf')
        self.stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


class Exp_Forecasting(Exp_Basic):

    def _get_data(self, flag: str):
        return data_provider(self.config, flag)

    def _select_optimizer(self):
        lr = self.config['training']['learning_rate']
        return optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )

    def _forward(self, batch):
        text_emb = None
        if len(batch) == 5:
            x_enc, x_dec, x_mark_enc, x_mark_dec, text_emb = batch
            text_emb = text_emb.float().to(self.device)
        else:
            x_enc, x_dec, x_mark_enc, x_mark_dec = batch

        x_enc      = x_enc.float().to(self.device)
        x_dec      = x_dec.float().to(self.device)
        x_mark_enc = x_mark_enc.float().to(self.device)
        x_mark_dec = x_mark_dec.float().to(self.device)

        use_amp = self.config['compute'].get('use_amp', False) and self.device.type == 'cuda'
        if use_amp:
            with torch.cuda.amp.autocast():
                out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, text_emb=text_emb)
        else:
            out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, text_emb=text_emb)
        return out, x_dec

    def vali(self, loader) -> float:
        self.model.eval()
        losses = []
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch in loader:
                pred, target = self._forward(batch)
                f_dim = -1 if self.config['data'].get('features') == 'MS' else 0
                pred   = pred[:, -self.config['model']['pred_len']:, f_dim:].float()
                target = target[:, -self.config['model']['pred_len']:, f_dim:].float().to(self.device)
                losses.append(criterion(pred, target).item())
        self.model.train()
        return float(np.mean(losses))

    def train(self, result_dir: str) -> dict:
        train_data, train_loader = self._get_data('train')
        _, val_loader = self._get_data('val')

        os.makedirs(result_dir, exist_ok=True)
        checkpoint_path = os.path.join(result_dir, 'checkpoint.pth')
        train_cfg = self.config['training']
        early_stopping = EarlyStopping(train_cfg['patience'], checkpoint_path)
        optimizer = self._select_optimizer()
        criterion = nn.MSELoss()

        use_amp = self.config['compute'].get('use_amp', False) and self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        self.model.train()
        epoch_bar = tqdm(range(train_cfg['train_epochs']), desc='Training', unit='epoch')
        for epoch in epoch_bar:
            train_losses = []
            batch_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False, unit='batch')
            for batch in batch_bar:
                optimizer.zero_grad()
                pred, target = self._forward(batch)

                f_dim = -1 if self.config['data'].get('features') == 'MS' else 0
                pred   = pred[:, -self.config['model']['pred_len']:, f_dim:].float()
                target = target[:, -self.config['model']['pred_len']:, f_dim:].float().to(self.device)

                loss = criterion(pred, target)
                train_losses.append(loss.item())

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                batch_bar.set_postfix(loss=f'{loss.item():.4f}')

            val_loss = self.vali(val_loader)
            train_loss = float(np.mean(train_losses))
            epoch_bar.set_postfix(train=f'{train_loss:.4f}', val=f'{val_loss:.4f}')

            early_stopping(val_loss, self.model)
            if early_stopping.stop:
                tqdm.write(f'Early stopping at epoch {epoch+1}')
                break

        # restore best weights
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        return {}

    def test(self, result_dir: str) -> dict:
        _, test_loader = self._get_data('test')
        self.model.eval()

        diag_cfg = self.config.get('diagnostics', {})
        diag_enabled = diag_cfg.get('enabled', False)
        log_alpha = diag_cfg.get('log_alpha', False)
        log_gates = diag_cfg.get('log_gates', False)
        log_beta  = diag_cfg.get('log_beta',  False)
        log_attn  = diag_cfg.get('log_attn',  False)

        preds, trues = [], []
        alpha_vals, gate_vals, beta_vals, attn_vals = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing', unit='batch'):
                pred, target = self._forward(batch)
                f_dim = -1 if self.config['data'].get('features') == 'MS' else 0
                pred   = pred[:, -self.config['model']['pred_len']:, f_dim:].float().cpu().numpy()
                target = target[:, -self.config['model']['pred_len']:, f_dim:].float().cpu().numpy()
                preds.append(pred)
                trues.append(target)

                if diag_enabled:
                    if log_alpha and hasattr(self.model, 'last_alpha'):
                        alpha_vals.append(float(self.model.last_alpha.mean().cpu()))
                    if log_gates and hasattr(self.model, 'last_gate'):
                        gate_vals.append(self.model.last_gate.cpu().numpy())
                    if log_beta and hasattr(self.model, 'last_beta'):
                        beta_vals.append(self.model.last_beta.cpu().numpy())
                    if log_attn and hasattr(self.model, 'last_attn_weights'):
                        attn_vals.append(self.model.last_attn_weights.mean(0).cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        metrics = dict(mae=float(mae), mse=float(mse),
                       rmse=float(rmse), mape=float(mape), mspe=float(mspe))
        print(f'Test | MAE={mae:.4f}  MSE={mse:.4f}  RMSE={rmse:.4f}')

        np.save(os.path.join(result_dir, 'preds.npy'), preds)
        np.save(os.path.join(result_dir, 'trues.npy'), trues)

        # ── Diagnostics ────────────────────────────────────────────────────
        if diag_enabled:
            diag = {}
            if alpha_vals:
                diag['alpha_mean'] = float(np.mean(alpha_vals))
                diag['alpha_std']  = float(np.std(alpha_vals))
            if gate_vals:
                gates = np.stack(gate_vals).mean(0)   # [d_model]
                diag['gate_mean'] = float(gates.mean())
                diag['gate_std']  = float(gates.std())
                diag['gate_per_dim_mean'] = gates.tolist()
            if beta_vals:
                betas = np.stack(beta_vals).mean(0)
                diag['beta_mean'] = float(betas.mean())
                diag['beta_std']  = float(betas.std())
            if attn_vals:
                attn = np.stack(attn_vals).mean(0)   # [num_patches, n_text_tokens]
                diag['attn_mean'] = float(attn.mean())
                diag['attn_per_patch'] = attn.mean(-1).tolist()
            metrics['diagnostics'] = diag

        return metrics
