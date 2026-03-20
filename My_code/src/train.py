import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.utils import get_data, save_model, load_model
from src.CBM_dataset import CBMDataset  
from src.models import MULT4Model       

def train_epoch(model, optimizer, criterion, loader, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for (idxs, snd, nF, fF, acc), ys, _ in loader:

        snd, nF, fF, acc, ys = [t.to(device) for t in (snd, nF, fF, acc, ys)]
        
        optimizer.zero_grad()
        preds, _ = model(snd, nF, fF, acc)   # preds: [B, C]
        loss = criterion(preds, ys)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * ys.size(0)
        all_preds .append(preds.argmax(dim=1).cpu())
        all_labels.append(ys.cpu())
    
    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average='macro')
    return avg_loss, acc, f1

def eval_epoch(model, criterion, loader, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for (idxs, snd, nF, fF, acc), ys, _ in loader:
            snd, nF, fF, acc, ys = [t.to(device) for t in (snd, nF, fF, acc, ys)]
            preds, _ = model(snd, nF, fF, acc)
            loss = criterion(preds, ys)
            
            total_loss += loss.item() * ys.size(0)
            all_preds .append(preds.argmax(dim=1).cpu())
            all_labels.append(ys.cpu())
    
    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, average='macro')
    return avg_loss, acc, f1

def main():
    # —— 1) 解析命令行 / 配置 —— 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',  type=str, default='/home/manager/lkl/AAAI26_Multimodal/dataset/m1474014/CBM_FinalDatabase')
    parser.add_argument('--model',      type=str, default='MULT4')
    parser.add_argument('--fold',       type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--epochs',     type=int, default=20)
    parser.add_argument('--use_cuda',   action='store_true')
    parser.add_argument('--add_noise',   action='store_true')
    parser.add_argument('--noise_snr',   type=float, default=5.0)
    parser.add_argument('--fill_missing',action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    # —— 2) 构造 Dataset & DataLoader —— 
    train_ds = get_data(args, split='train')
    test_ds  = get_data(args, split='test')

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=CBMDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=CBMDataset.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # —— 3) 构造模型 —— 
    # 把原始 MFCC 维度告诉模型
    # CBMDataset 输出的是 [B,T,MFCC_DIM]，所以 orig_d_* = MFCC_DIM
    args.orig_d_s   = 24
    args.orig_d_nF  = 24
    args.orig_d_fF  = 24
    args.orig_d_acc = 24
    # 还要把类别数告诉模型
    args.output_dim = len(train_ds.dataset.class2idx)

    model = MULT4Model(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    best_val_f1 = 0.0

    # —— 4) 训练循环 —— 
    for epoch in range(1, args.epochs+1):
        start = time.time()
        train_loss, train_acc, train_f1 = train_epoch(model, optimizer, criterion, train_loader, device)
        val_loss,   val_acc,   val_f1   = eval_epoch(model, criterion, test_loader,  device)

        scheduler.step(val_loss)
        elapsed = time.time() - start

        print(f"[Epoch {epoch:02d}] "
              f"Time {elapsed:.1f}s  "
              f"TrLoss {train_loss:.4f}  TrF1 {train_f1:.4f}  "
              f"VaLoss {val_loss:.4f}  VaF1 {val_f1:.4f}")

        # 保存最优
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_model(args, model, name='best')

    # —— 5) 测试 & 报告 —— 
    print("=== Final Evaluation on Test Fold ===")
    test_loss, test_acc, test_f1 = eval_epoch(model, criterion, test_loader, device)
    print(f"Test Loss {test_loss:.4f}  Test F1 {test_f1:.4f}")

if __name__ == '__main__':
    main()
