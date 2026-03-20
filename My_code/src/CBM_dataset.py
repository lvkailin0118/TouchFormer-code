############################################### 粗粒度分类 ######################################################
import os, math, random, pickle, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from scipy.io import loadmat
from python_speech_features import mfcc

# ---------- 常量 ----------
REQUIRED_MODALITIES = ['sound', 'normalForce', 'frictionForce', 'accelDFT']
SAMPLE_RATE = {'sound': 44100, 'normalForce': 3000, 'frictionForce': 3000, 'accelDFT': 3000}
MFCC_DIM, NFFT = 24, 1103
GLASS_CID = 'C5'


NOT_RECORDED_YET = set([
    'C2_S1_M5', 'C2_S1_M6', 'C3_S1_M4', 'C3_S1_M13', 'C3_S1_M14', 'C3_S1_M15',
    'C3_S1_M16', 'C3_S1_M17', 'C3_S1_M18', 'C3_S1_M20', 'C3_S1_M22', 'C3_S1_M23',
    'C3_S1_M24', 'C3_S1_M25', 'C3_S2_M1', 'C3_S2_M3', 'C3_S2_M5',
    'C4_S1_M4', 'C4_S1_M5', 'C4_S1_M6', 'C4_S3_M2',
    'C5_S1_M2', 'C5_S1_M3', 'C5_S1_M4', 'C5_S1_M5', 'C5_S1_M6', 'C5_S1_M7',
    'C5_S2_M1',
    'C6_S1_M4', 'C6_S1_M5', 'C6_S2_M4',
    'C6_S3_M8', 'C6_S3_M9', 'C6_S3_M10', 'C6_S3_M11', 'C6_S3_M12',
    'C6_S4_M2', 'C6_S5_M1', 'C6_S5_M2', 'C6_S5_M3', 'C6_S6_M1', 'C6_S6_M2',
    'C7_S2_M5', 'C7_S2_M6', 'C7_S2_M7', 'C7_S2_M8', 'C7_S3_M4', 'C7_S3_M5',
    'C7_S3_M6', 'C7_S5_M1', 'C8_S3_M2', 'C8_S4_M2'
])

# ---------- 工具函数 ----------
def _mfcc_feat(x: np.ndarray, sr: int) -> torch.Tensor:
    return torch.tensor(mfcc(x, sr, numcep=MFCC_DIM, nfft=NFFT), dtype=torch.float32)

def _add_white_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    if snr_db <= 0: return x
    noise = np.random.randn(*x.shape)
    p_sig = (x**2).mean()
    noise = noise/noise.std() * math.sqrt(p_sig / 10**(snr_db/10))
    return x + noise

def _cleanup_paths(paths: List[str]) -> List[str]:
    """过滤 tmp 文件和官方未录制材料"""
    keep = []
    for p in paths:
        mat_name = "_".join(Path(p).parent.parts[-3:])          # e.g. C3_S1_M12
        file_name = Path(p).name.lower()
        if mat_name in NOT_RECORDED_YET or 'tmp' in file_name:
            continue
        keep.append(p)
    return keep

# ---------- 数据集 ----------
class CBMDataset(Dataset):
    def __init__(self, root_dir: str,
                 add_noise: bool = False, noise_snr: float = 5.0,
                 add_robust_noise: bool = False, robust_noise_ratio: float = 0.2,
                 robust_noise_strength: float = 10,
                 fill_missing: bool = False, verbose: bool = True,
                 use_cache: bool = True, train: bool = True):
        super().__init__()
        self.add_noise = add_noise
        self.noise_snr = noise_snr
        self.add_robust_noise = add_robust_noise
        self.robust_noise_ratio = robust_noise_ratio
        self.robust_noise_strength = robust_noise_strength
        self.train = train
        cache_path = Path(root_dir) / 'cbm_dataset_cache.pkl'

        if use_cache and cache_path.exists():
            if verbose: print("[CBM] Loading cache...")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.records       = data['records']
            self.material_ids  = data['material_ids']
            self.classes       = data['classes']
            self.class2idx     = data['class2idx']
        else:
            if verbose: print("[CBM] Building cache...")
            self.records, self.material_ids = self._prepare_data(root_dir, fill_missing, verbose)
            self.classes   = sorted({mid.split('_')[0] for mid in self.material_ids})
            self.class2idx = {c: i for i, c in enumerate(self.classes)}
            with open(cache_path, 'wb') as f:
                pickle.dump({'records': self.records,
                             'material_ids': self.material_ids,
                             'classes': self.classes,
                             'class2idx': self.class2idx}, f)

    # -------- 私有 --------
    def _prepare_data(self, root_dir, fill_missing, verbose):
        records, material_ids = [], []
        raw_paths = [os.path.join(dp, f)
                     for dp, _, fs in os.walk(root_dir)
                     for f in fs if f.endswith('.mat')]
        paths = _cleanup_paths(raw_paths)

        for path in paths:
            try:
                mat = loadmat(path, struct_as_record=False, squeeze_me=True)
                if 'finalMaterialRecording' not in mat:
                    if verbose: print(f"  [skip] no key: {path}")
                    continue
                rec = mat['finalMaterialRecording']
                if isinstance(rec, np.ndarray): rec = rec[0]
            except Exception as e:
                if verbose: print(f"  [skip] invalid mat: {path} ({e})")
                continue

            feats = []
            for m in REQUIRED_MODALITIES:
                if hasattr(rec, m) and getattr(rec, m).size > 0:
                    feat_np = getattr(rec, m).squeeze().astype(np.float32)
                    feats.append(_mfcc_feat(feat_np, SAMPLE_RATE[m]))
                elif fill_missing:
                    feats.append(torch.zeros(1, MFCC_DIM))
                else:
                    break          # 该记录缺模态且不允许填充
            else:
                records.append(feats)
                material_ids.append('_'.join(Path(path).parts[-4:-1]))

        if verbose:
            print(f"[CBM] usable records {len(records)}/{len(paths)}")
        return records, material_ids

    # -------- Dataset API --------
    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        feats = [f.clone() for f in self.records[idx]]

        # 1) 模拟“模态崩溃”——仅在 train/test 指定开启时
        if self.add_robust_noise and random.random() < self.robust_noise_ratio:
            m_idx = random.randint(0, 3)
            feats[m_idx] = torch.randn_like(feats[m_idx]) * self.robust_noise_strength

        # 2) 轻度白噪声——仅在 add_noise=True 时
        if self.add_noise:
            feats = [torch.tensor(_add_white_noise(f.numpy(), self.noise_snr), dtype=torch.float32)
                     for f in feats]

        class_id = self.material_ids[idx].split('_')[0]
        y = torch.tensor(self.class2idx[class_id])
        return (idx, *feats), y, None

    # -------- collate & split --------
    @staticmethod
    def _pad(seqs): return pad_sequence(seqs, batch_first=True)

    @staticmethod
    def collate_fn(batch):
        idxs, snd, nF, fF, acc, ys = zip(*[(b[0][0], b[0][1], b[0][2],
                                            b[0][3], b[0][4], b[1]) for b in batch])
        snd, nF, fF, acc = map(CBMDataset._pad, (snd, nF, fF, acc))
        return (torch.tensor(idxs), snd, nF, fF, acc), torch.stack(ys), None

    def split_five_fold(self, fold_idx=0) -> Tuple[Subset, Subset]:
        tr, te = [], []
        for mid in sorted(set(self.material_ids)):
            ids = [i for i, m in enumerate(self.material_ids) if m == mid]
            if mid.startswith(GLASS_CID):               # 玻璃全部进训练
                tr.extend(ids)
            elif len(ids) == 5:                         # 常规 5-fold
                te.append(ids[fold_idx])
                tr.extend(i for i in ids if i != ids[fold_idx])
        return Subset(self, tr), Subset(self, te)

# ---------- 简单测试 ----------
if __name__ == "__main__":
    root = "/path/to/CBM_FinalDatabase"
    ds = CBMDataset(root, use_cache=True, verbose=True)
    tr, _ = ds.split_five_fold(0)
    from torch.utils.data import DataLoader
    dl = DataLoader(tr, batch_size=4, shuffle=True, collate_fn=CBMDataset.collate_fn)
    (idx, snd, nF, fF, acc), y, _ = next(iter(dl))
    print("Batch shapes:", snd.shape, nF.shape, fF.shape, acc.shape, y.shape)



#################################################### subclass分类 #################################################

import os, math, random, pickle, warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence
from scipy.io import loadmat
from python_speech_features import mfcc

# ---------- 常量 ----------
REQUIRED_MODALITIES = ['sound', 'normalForce', 'frictionForce', 'accelDFT']
SAMPLE_RATE = {'sound': 44100, 'normalForce': 3000, 'frictionForce': 3000, 'accelDFT': 3000}
MFCC_DIM, NFFT = 24, 1103
GLASS_CID = 'C5'

# 未录制材料集合
NOT_RECORDED_YET = set([
    'C2_S1_M5', 'C2_S1_M6', 'C3_S1_M4', 'C3_S1_M13', 'C3_S1_M14', 'C3_S1_M15',
    'C3_S1_M16', 'C3_S1_M17', 'C3_S1_M18', 'C3_S1_M20', 'C3_S1_M22', 'C3_S1_M23',
    'C3_S1_M24', 'C3_S1_M25', 'C3_S2_M1', 'C3_S2_M3', 'C3_S2_M5',
    'C4_S1_M4', 'C4_S1_M5', 'C4_S1_M6', 'C4_S3_M2',
    'C5_S1_M2', 'C5_S1_M3', 'C5_S1_M4', 'C5_S1_M5', 'C5_S1_M6', 'C5_S1_M7',
    'C5_S2_M1', 'C6_S1_M4', 'C6_S1_M5', 'C6_S2_M4',
    'C6_S3_M8', 'C6_S3_M9', 'C6_S3_M10', 'C6_S3_M11', 'C6_S3_M12',
    'C6_S4_M2', 'C6_S5_M1', 'C6_S5_M2', 'C6_S5_M3', 'C6_S6_M1', 'C6_S6_M2',
    'C7_S2_M5', 'C7_S2_M6', 'C7_S2_M7', 'C7_S2_M8', 'C7_S3_M4', 'C7_S3_M5',
    'C7_S3_M6', 'C7_S5_M1', 'C8_S3_M2', 'C8_S4_M2'
])

# ---------- 工具函数 ----------
def _mfcc_feat(x: np.ndarray, sr: int) -> torch.Tensor:
    return torch.tensor(mfcc(x, sr, numcep=MFCC_DIM, nfft=NFFT), dtype=torch.float32)

def _add_white_noise(x: np.ndarray, snr_db: float) -> np.ndarray:
    if snr_db <= 0: return x
    noise = np.random.randn(*x.shape)
    p_sig = (x**2).mean()
    noise = noise/noise.std() * math.sqrt(p_sig / 10**(snr_db/10))
    return x + noise

def _cleanup_paths(paths: List[str]) -> List[str]:
    keep = []
    for p in paths:
        mat_name = "_".join(Path(p).parent.parts[-3:])
        if mat_name in NOT_RECORDED_YET or 'tmp' in Path(p).name.lower():
            continue
        keep.append(p)
    return keep

# ---------- 数据集 ----------
class CBMDataset(Dataset):
    def __init__(self, root_dir: str,
                 add_noise: bool = False, noise_snr: float = 5.0,
                 add_robust_noise: bool = False, robust_noise_ratio: float = 0.2,
                 robust_noise_strength: float = 10,
                 fill_missing: bool = False, verbose: bool = True,
                 use_cache: bool = True, train: bool = True):
        super().__init__()
        self.add_noise = add_noise
        self.noise_snr = noise_snr
        self.add_robust_noise = add_robust_noise
        self.robust_noise_ratio = robust_noise_ratio
        self.robust_noise_strength = robust_noise_strength
        self.train = train
        cache_path = Path(root_dir) / 'cbm_dataset_cache_fine.pkl'

        if use_cache and cache_path.exists():
            if verbose: print("[CBM-Fine] Loading cache...")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            self.records = data['records']
            self.material_ids = data['material_ids']
            self.classes = data['classes']
            self.class2idx = data['class2idx']
        else:
            if verbose: print("[CBM-Fine] Building cache...")
            self.records, self.material_ids = self._prepare_data(root_dir, fill_missing, verbose)
            self.classes = sorted({mid.split('_')[0] + '_' + mid.split('_')[1] for mid in self.material_ids})
            self.class2idx = {c: i for i, c in enumerate(self.classes)}
            with open(cache_path, 'wb') as f:
                pickle.dump({'records': self.records,
                             'material_ids': self.material_ids,
                             'classes': self.classes,
                             'class2idx': self.class2idx}, f)

    def _prepare_data(self, root_dir, fill_missing, verbose):
        records, material_ids = [], []
        raw_paths = [os.path.join(dp, f)
                     for dp, _, fs in os.walk(root_dir)
                     for f in fs if f.endswith('.mat')]
        paths = _cleanup_paths(raw_paths)

        for path in paths:
            mat = loadmat(path, struct_as_record=False, squeeze_me=True)
            if 'finalMaterialRecording' not in mat:
                continue
            rec = mat['finalMaterialRecording']
            if isinstance(rec, np.ndarray): rec = rec[0]

            feats = []
            for m in REQUIRED_MODALITIES:
                arr = getattr(rec, m, np.array([]))
                if arr.size > 0:
                    feats.append(_mfcc_feat(arr.squeeze(), SAMPLE_RATE[m]))
                elif fill_missing:
                    feats.append(torch.zeros(1, MFCC_DIM))
                else:
                    break
            else:
                records.append(feats)
                material_ids.append('_'.join(Path(path).parts[-4:-1]))

        if verbose:
            print(f"[CBM-Fine] usable records {len(records)}/{len(paths)}")
        return records, material_ids

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        feats = [f.clone() for f in self.records[idx]]

        if self.add_robust_noise and random.random() < self.robust_noise_ratio:
            m_idx = random.randint(0, 3)
            feats[m_idx] = torch.randn_like(feats[m_idx]) * self.robust_noise_strength

        if self.add_noise:
            feats = [torch.tensor(_add_white_noise(f.numpy(), self.noise_snr), dtype=torch.float32) for f in feats]

        label_key = '_'.join(self.material_ids[idx].split('_')[:2])
        y = torch.tensor(self.class2idx[label_key])

        return (idx, *feats), y, None

