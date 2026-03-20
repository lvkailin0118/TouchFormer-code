import torch
import os
from src.CBM_dataset import CBMDataset

def get_data(args, split: str = 'train'):
    # 构造完整 Dataset
    full_ds = CBMDataset(
        root_dir      = args.data_path,
        add_noise     = getattr(args, 'add_noise', False),
        noise_snr     = getattr(args, 'noise_snr', 5.0),
        fill_missing  = getattr(args, 'fill_missing', False),
        verbose       = False
    )
    # 划分五折
    train_ds, test_ds = full_ds.split_five_fold(args.fold)
    return train_ds if split == 'train' else test_ds


def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model: torch.nn.Module, name: str = ''):
    fname = save_load_name(args, name) + ".pt"
    os.makedirs("pre_trained_models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("pre_trained_models", fname))

def load_model(args, model_cls, name: str = '') -> torch.nn.Module:
    fname = save_load_name(args, name) + ".pt"
    ckpt = torch.load(os.path.join("pre_trained_models", fname))
    model = model_cls(args) 
    model.load_state_dict(ckpt)
    return model

