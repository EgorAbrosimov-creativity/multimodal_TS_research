"""
compare_results.py — print a table of all completed experiments.

Usage:
    python compare_results.py
    python compare_results.py --filter model=PatchTST
    python compare_results.py --filter dataset=ETTh1 --sort mse
"""

import argparse
import json
from pathlib import Path

REGISTRY_PATH = Path('experiments/registry.json')

METRIC_COLS = ['mae', 'mse', 'rmse']


def load_registry() -> list[dict]:
    if not REGISTRY_PATH.exists():
        print('No experiments found. Run some experiments first.')
        return []
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def apply_filter(records: list[dict], filters: list[str]) -> list[dict]:
    for f in filters:
        key, _, value = f.partition('=')
        key = key.strip()
        records = [
            r for r in records
            if str(r.get(key, r['metrics'].get(key, ''))) == value
        ]
    return records


def keep_latest(records: list[dict]) -> list[dict]:
    """Keep only the most recent run per experiment name."""
    seen: dict[str, dict] = {}
    for r in records:
        name = r['name']
        if name not in seen or r['timestamp'] > seen[name]['timestamp']:
            seen[name] = r
    return list(seen.values())


def print_table(records: list[dict], sort_by: str | None):
    if not records:
        print('No matching experiments.')
        return

    if sort_by and sort_by in METRIC_COLS:
        records = sorted(records, key=lambda r: r['metrics'].get(sort_by, float('inf')))

    # Column widths
    name_w    = max(len(r['name']) for r in records) + 2
    model_w   = max(len(r['model']) for r in records) + 2
    dataset_w = max(len(r['dataset']) for r in records) + 2

    header = (
        f"{'name':<{name_w}} "
        f"{'model':<{model_w}} "
        f"{'dataset':<{dataset_w}} "
        f"{'pred_len':>8}  "
        f"{'frac':>6}  "
        f"{'MAE':>8}  "
        f"{'MSE':>8}  "
        f"{'RMSE':>8}  "
        f"{'timestamp'}"
    )
    sep = '-' * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in records:
        m = r.get('metrics', {})
        frac = r.get('train_fraction', 1.0)
        print(
            f"{r['name']:<{name_w}} "
            f"{r['model']:<{model_w}} "
            f"{r['dataset']:<{dataset_w}} "
            f"{r['pred_len']:>8}  "
            f"{frac:>5.0%}  "
            f"{m.get('mae', float('nan')):>8.4f}  "
            f"{m.get('mse', float('nan')):>8.4f}  "
            f"{m.get('rmse', float('nan')):>8.4f}  "
            f"{r.get('timestamp', '')}"
        )
    print(sep)
    print(f'{len(records)} experiment(s)')


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results.')
    parser.add_argument('--filter', nargs='*', default=[],
                        metavar='KEY=VALUE',
                        help='Filter experiments, e.g. --filter model=PatchTST dataset=ETTh1')
    parser.add_argument('--sort', default=None,
                        choices=METRIC_COLS,
                        help='Sort by metric (ascending)')
    args = parser.parse_args()

    records = load_registry()
    records = keep_latest(records)
    records = apply_filter(records, args.filter)
    print_table(records, args.sort)


if __name__ == '__main__':
    main()
