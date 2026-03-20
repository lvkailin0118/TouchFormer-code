import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from scipy.io import loadmat
from python_speech_features import mfcc
import os

SAMPLE_RATE = {
    'sound': 44100,
    'normalForce': 100,
    'frictionForce': 100,
    'accelDFT': 500
}

MFCC_DIM = 13
NFFT = {
    'sound': 2048,
    'normalForce': 64,
    'frictionForce': 64,
    'accelDFT': 256
}

REQUIRED_MODALITIES = ['sound', 'normalForce', 'frictionForce', 'accelDFT']

class CustomDataset(Dataset):
    def __init__(self, root_dir, verbose=True):
        super().__init__()
        self.records, self.material_ids = self._prepare_data(root_dir, verbose)
        self.classes = sorted(set(self.material_ids))
        self.class2idx = {cls: i for i, cls in enumerate(self.classes)}

    def _mfcc_feat(self, x, sr, modality):
        return torch.tensor(
            mfcc(x, samplerate=sr, numcep=MFCC_DIM, nfft=NFFT[modality]),
            dtype=torch.float32
        )

    ####################################粗分类####################################
    # def _prepare_data(self, root_dir, verbose):
    #     records, material_ids = [], []
    #     paths = [os.path.join(dp, f)
    #             for dp, _, fs in os.walk(root_dir)
    #             for f in fs if f.endswith('.mat')]

    #     for path in paths:
    #         mat = loadmat(path, struct_as_record=False, squeeze_me=True)

    #         feats = []
    #         skip_file = False
    #         for m in REQUIRED_MODALITIES:
    #             if m in mat and mat[m].size > 0:
    #                 feat_np = mat[m].squeeze().astype(np.float32)
    #                 feats.append(self._mfcc_feat(feat_np, SAMPLE_RATE[m], m))
    #             else:
    #                 skip_file = True
    #                 if verbose:
    #                     print(f"Skipping {path}, missing {m}")
    #                 break

    #         if skip_file:
    #             continue

    #         # 使用上层目录名作为类别（粗分类）
    #         material_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
    #         records.append(feats)
    #         material_ids.append(material_name)

    #     if verbose:
    #         print(f"[CustomDataset] Loaded {len(records)} records from {len(paths)} files.")
    #     return records, material_ids

    ####################################细分类####################################
    def _prepare_data(self, root_dir, verbose):
        records, material_ids = [], []
        paths = [os.path.join(dp, f)
                for dp, _, fs in os.walk(root_dir)
                for f in fs if f.endswith('.mat')]

        for path in paths:
            mat = loadmat(path, struct_as_record=False, squeeze_me=True)

            feats = []
            skip_file = False
            for m in REQUIRED_MODALITIES:
                if m in mat and mat[m].size > 0:
                    feat_np = mat[m].squeeze().astype(np.float32)
                    feats.append(self._mfcc_feat(feat_np, SAMPLE_RATE[m], m))
                else:
                    skip_file = True
                    if verbose:
                        print(f"Skipping {path}, missing {m}")
                    break

            if skip_file:
                continue

            # 提取第二级子目录作为类别名
            sub_category = os.path.basename(os.path.dirname(path))
            main_category = os.path.basename(os.path.dirname(os.path.dirname(path)))

            if main_category in ['mental', 'paper', 'polymer', 'rubber', 'stone', 'textiles', 'wood']:
                records.append(feats)
                material_ids.append(sub_category)

        if verbose:
            print(f"[CustomDataset] Loaded {len(records)} records from {len(paths)} files.")
            print(f"Classes found: {sorted(set(material_ids))}")

        return records, material_ids

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        feats = [f.clone() for f in self.records[idx]]
        class_id = self.material_ids[idx]
        y = torch.tensor(self.class2idx[class_id])
        return (idx, *feats), y

    @staticmethod
    def collate_fn(batch):
        idxs, snd, nF, fF, acc, ys = zip(*[(b[0][0], b[0][1], b[0][2], b[0][3], b[0][4], b[1]) for b in batch])
        snd, nF, fF, acc = map(lambda x: pad_sequence(x, batch_first=True), (snd, nF, fF, acc))
        return (torch.tensor(idxs), snd, nF, fF, acc), torch.stack(ys)
