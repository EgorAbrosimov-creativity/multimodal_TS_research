"""
encode_descriptions.py — offline pre-encoding of text descriptions.

Generates template-based descriptions for every sliding window of a dataset split,
encodes them with MiniLM-L6, and saves a .npy file of shape [N, 384] — one
embedding per window (indexed by s_begin, matching Dataset.__getitem__).

Run once per dataset/split before training. The resulting .npy paths go into
the YAML config under data.text_emb_path (for train split); val/test paths are
derived by replacing _train_ with _val_ / _test_.

Usage:
    python encode_descriptions.py \\
        --dataset ETTh1 \\
        --split train \\
        --out /content/drive/MyDrive/multimodal_TS_research/iteration_2/embeddings/ETTh1_train_minilm.npy

    # All splits at once:
    for split in train val test; do
        python encode_descriptions.py --dataset ETTh1 --split $split \\
            --out /content/drive/.../ETTh1_${split}_minilm.npy
    done

Arguments:
    --dataset   ETTh1 | ETTh2 | ETTm1 | ETTm2 | custom
    --split     train | val | test
    --out       path for the output .npy file
    --seq_len   input window length (default: 512)
    --label_len decoder prefix length (default: 48)
    --pred_len  forecast horizon — used only in description text (default: 96)
    --root_path root directory of the dataset CSVs (default: ./dataset/ETT-small/)
    --data_path CSV filename (default: ETTh1.csv)
    --freq      h | t (default: h)
    --features  M | MS | S (default: M)
    --model     HuggingFace model name for encoding (default: sentence-transformers/all-MiniLM-L6-v2)
    --batch_sz  encoding batch size (default: 512)
    --std_p80   pre-computed 80th percentile of std across train windows (optional)
    --slope_p80 pre-computed 80th percentile of |slope| across train windows (optional)
"""

import argparse
import os
import sys

# Ensure repo root is on sys.path regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_provider.data_loader import (
    Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom,
)
from layers.TextEncoder import TextEncoder, generate_ts_description

DATA_DICT = {
    'ETTh1': (Dataset_ETT_hour,   'ETTh1.csv', 'h'),
    'ETTh2': (Dataset_ETT_hour,   'ETTh2.csv', 'h'),
    'ETTm1': (Dataset_ETT_minute, 'ETTm1.csv', 't'),
    'ETTm2': (Dataset_ETT_minute, 'ETTm2.csv', 't'),
    'custom': (Dataset_Custom,    'data.csv',  'h'),
}


def compute_dataset_stats(dataset) -> dict:
    """
    Compute per-window slope and std for all train windows, then return
    the 80th percentile of each — used for relative language in descriptions.
    """
    stds, slopes = [], []
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    for batch in tqdm(loader, desc='Computing dataset stats', leave=False):
        x_enc = batch[0].float()           # [B, T, n_vars]
        series = x_enc.mean(dim=-1).numpy()  # [B, T]
        T = series.shape[1]
        stds.extend(np.std(series, axis=1).tolist())
        slopes.extend(((series[:, -1] - series[:, 0]) / max(T - 1, 1)).tolist())
    return {
        'std_p80':   float(np.percentile(stds,         80)),
        'slope_p80': float(np.percentile(np.abs(slopes), 80)),
    }


def encode_split(
    dataset,
    encoder: TextEncoder,
    dataset_name: str,
    pred_len: int,
    dataset_stats: dict | None,
    batch_sz: int,
) -> np.ndarray:
    """Encode all windows and return embeddings array of shape [N, hidden_dim]."""
    loader = DataLoader(dataset, batch_size=batch_sz, shuffle=False, num_workers=0)
    all_embs = []
    for batch in tqdm(loader, desc='Encoding', unit='batch'):
        x_enc      = batch[0].float()   # [B, T, n_vars]
        x_mark_enc = batch[2].float()   # [B, T, F]
        texts = generate_ts_description(
            x_enc, dataset_name, pred_len, x_mark_enc, dataset_stats)
        with torch.no_grad():
            embs = encoder(texts)       # [B, hidden_dim]
        all_embs.append(embs.cpu().numpy())
    return np.concatenate(all_embs, axis=0)  # [N, hidden_dim]


def main():
    parser = argparse.ArgumentParser(description='Offline text embedding pre-encoding.')
    parser.add_argument('--dataset',   required=True, choices=list(DATA_DICT.keys()))
    parser.add_argument('--split',     required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--out',       required=True, help='Output .npy file path')
    parser.add_argument('--seq_len',   type=int, default=512)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len',  type=int, default=96)
    parser.add_argument('--root_path', default='./dataset/ETT-small/')
    parser.add_argument('--data_path', default=None, help='CSV filename (default: auto from dataset)')
    parser.add_argument('--freq',      default=None, help='h or t (default: auto from dataset)')
    parser.add_argument('--features',  default='M')
    parser.add_argument('--model',     default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--batch_sz',  type=int, default=512)
    parser.add_argument('--std_p80',   type=float, default=None,
                        help='80th percentile of std (computed from train if omitted for train split)')
    parser.add_argument('--slope_p80', type=float, default=None,
                        help='80th percentile of |slope|')
    args = parser.parse_args()

    DatasetClass, default_csv, default_freq = DATA_DICT[args.dataset]
    data_path = args.data_path or default_csv
    freq      = args.freq      or default_freq

    print(f'\nDataset : {args.dataset}  split={args.split}')
    print(f'Encoder : {args.model}')
    print(f'Output  : {args.out}\n')

    # ── Build dataset ─────────────────────────────────────────────────────
    dataset = DatasetClass(
        root_path=args.root_path,
        flag=args.split,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        data_path=data_path,
        scale=True,
        timeenc=1,
        freq=freq,
    )
    print(f'Windows in split: {len(dataset)}')

    # ── Dataset-level stats (from train) ─────────────────────────────────
    dataset_stats = None
    if args.std_p80 is not None and args.slope_p80 is not None:
        dataset_stats = {'std_p80': args.std_p80, 'slope_p80': args.slope_p80}
        print(f'Using provided stats: std_p80={args.std_p80:.4f}, slope_p80={args.slope_p80:.4f}')
    elif args.split == 'train':
        print('Computing dataset stats from train split...')
        dataset_stats = compute_dataset_stats(dataset)
        print(f'  std_p80={dataset_stats["std_p80"]:.4f}  '
              f'slope_p80={dataset_stats["slope_p80"]:.4f}')
        print('  (pass --std_p80 and --slope_p80 to reuse these for val/test)')
    else:
        print('No dataset stats provided — relative language disabled. '
              'Re-run val/test with --std_p80 and --slope_p80 from the train run.')

    # ── Load encoder ──────────────────────────────────────────────────────
    print(f'\nLoading {args.model}...')
    encoder = TextEncoder(model_name=args.model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if hasattr(encoder, '_model'):
        encoder._model = encoder._model.to(device)
    print(f'Using device: {device}')

    # ── Encode ────────────────────────────────────────────────────────────
    embs = encode_split(dataset, encoder, args.dataset, args.pred_len, dataset_stats, args.batch_sz)
    print(f'\nEmbeddings shape: {embs.shape}   dtype: {embs.dtype}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, embs)
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
