import platform
from torch.utils.data import DataLoader, Subset
from utils.data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

DATA_DICT = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}


def data_provider(config, flag):
    data_cfg = config['data']
    train_cfg = config['training']
    compute_cfg = config['compute']
    model_cfg = config['model']

    dataset_key = data_cfg['dataset']
    if dataset_key not in DATA_DICT:
        raise ValueError(
            f"Unknown dataset '{dataset_key}'. "
            f"Supported: {list(DATA_DICT.keys())}"
        )
    DatasetClass = DATA_DICT[dataset_key]

    shuffle = flag == 'train'
    train_fraction = train_cfg.get('train_fraction', 1.0)

    # Derive split-specific embedding path by replacing '_train_' in the base path
    base_emb_path = data_cfg.get('text_emb_path', None)
    if base_emb_path and flag != 'train':
        base_emb_path = base_emb_path.replace('_train_', f'_{flag}_')

    dataset = DatasetClass(
        root_path=data_cfg['root_path'],
        flag=flag,
        size=[model_cfg['seq_len'], model_cfg['label_len'], model_cfg['pred_len']],
        features=data_cfg.get('features', 'M'),
        data_path=data_cfg['data_path'],
        target=data_cfg.get('target', 'OT'),
        scale=True,
        timeenc=1,
        freq=data_cfg.get('freq', 'h'),
        text_emb_path=base_emb_path,
    )

    # Apply train_fraction: keep the MOST RECENT fraction of training windows
    # (cut the earlier part so training data is closest in time to validation/test)
    if flag == 'train' and train_fraction < 1.0:
        total = len(dataset)
        n = max(1, int(total * train_fraction))
        dataset = Subset(dataset, range(total - n, total))

    num_workers = compute_cfg.get('num_workers', 4)
    if platform.system() == 'Darwin':
        num_workers = 0  # macOS uses spawn, not fork — worker overhead exceeds benefit for small datasets

    loader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )

    return dataset, loader
