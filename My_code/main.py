import torch
import os
import time
import argparse
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
from src.utils import save_model
from src.CBM_dataset import CBMDataset
from src.models import MULT4Model
import torch.nn as nn
import torch.optim as optim

# def train_epoch(model, optimizer, criterion, loader, device):
#     model.train()
#     total_loss, all_preds, all_labels = 0, [], []
#     for (_, snd, nF, fF, acc), ys, _ in loader:
#         snd, nF, fF, acc, ys = [t.to(device) for t in (snd, nF, fF, acc, ys)]
#         optimizer.zero_grad()
#         preds, _ = model(snd, nF, fF, acc)
#         loss = criterion(preds, ys)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * ys.size(0)
#         all_preds.append(preds.argmax(dim=1).cpu())
#         all_labels.append(ys.cpu())

#     avg_loss = total_loss / len(loader.dataset)
#     preds = torch.cat(all_preds)
#     labels = torch.cat(all_labels)
#     acc = accuracy_score(labels, preds)
#     f1 = f1_score(labels, preds, average='macro')
#     return avg_loss, acc, f1

def train_epoch(model, optimizer, criterion, loader, device, lambda_contrast=0.1, temperature=0.2):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for (_, snd, nF, fF, acc), ys, _ in loader:
        snd, nF, fF, acc, ys = [t.to(device) for t in (snd, nF, fF, acc, ys)]
        optimizer.zero_grad()
        logits, embeddings_norm = model(snd, nF, fF, acc)

        cls_loss = criterion(logits, ys)

        # 对比学习损失
        sim_matrix = embeddings_norm @ embeddings_norm.T
        labels_eq = ys.unsqueeze(1) == ys.unsqueeze(0)
        mask_self = torch.eye(ys.size(0), device=device).bool()
        labels_eq = labels_eq.masked_fill(mask_self, False)

        exp_sim = torch.exp(sim_matrix / temperature)
        exp_sim = exp_sim.masked_fill(mask_self, 0)

        pos_sum = (exp_sim * labels_eq).sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        
        contrast_loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8)).mean()

        loss = cls_loss + lambda_contrast * contrast_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * ys.size(0)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(ys.cpu())

    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')

    return avg_loss, acc, f1

def eval_epoch(model, criterion, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for (_, snd, nF, fF, acc), ys, _ in loader:
            snd, nF, fF, acc, ys = [t.to(device) for t in (snd, nF, fF, acc, ys)]
            preds, _ = model(snd, nF, fF, acc)
            loss = criterion(preds, ys)
            total_loss += loss.item() * ys.size(0)
            all_preds.append(preds.argmax(dim=1).cpu())
            all_labels.append(ys.cpu())

    avg_loss = total_loss / len(loader.dataset)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return avg_loss, acc, f1

def main():
    parser = argparse.ArgumentParser()
     # 数据相关
    parser.add_argument('--data_path', type=str, default='/home/manager/lkl/AAAI26_Multimodal/dataset/m1474014/CBM_FinalDatabase')
    parser.add_argument('--fold',      type=int, default=0, help='五折索引 0-4')
    parser.add_argument('--use_cache', action='store_true', help='使用 pickle 缓存')
    parser.add_argument('--fill_missing', action='store_true', help='缺模态是否用 0 填充')

    # 噪声相关
    parser.add_argument('--add_noise', action='store_true', help='轻度白噪声增强')
    parser.add_argument('--noise_snr', type=float, default=5.0)
    parser.add_argument('--robust_noise_ratio', type=float, default=0.2)
    parser.add_argument('--robust_noise_strength', type=float, default=5.0)

    # 训练超参
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--use_cuda',   action='store_true')

    # 模型结构超参
    parser.add_argument('--num_heads',     type=int, default=4)
    parser.add_argument('--layers',        type=int, default=3)
    parser.add_argument('--attn_dropout',  type=float, default=0.1)
    parser.add_argument('--relu_dropout',  type=float, default=0.1)
    parser.add_argument('--embed_dropout', type=float, default=0.25)
    parser.add_argument('--res_dropout',   type=float, default=0.1)
    parser.add_argument('--out_dropout',   type=float, default=0.1)
    parser.add_argument('--attn_mask',     action='store_true')

    parser.add_argument('--model', type=str, default='MULT4')
    parser.add_argument('--aligned', action='store_true', help='模型是否对齐')
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    # 加载数据集并按 fold 划分
    train_ds_full = CBMDataset(args.data_path, add_noise=True, noise_snr=args.noise_snr,
                        add_robust_noise=True, robust_noise_ratio=0,
                        use_cache=args.use_cache, train=True)
    test_ds_full = CBMDataset(args.data_path, add_noise=False,
                        add_robust_noise=True, robust_noise_ratio=0,
                        use_cache=args.use_cache, train=False)

    train_ds, _ = train_ds_full.split_five_fold(args.fold)
    _, test_ds = test_ds_full.split_five_fold(args.fold)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=CBMDataset.collate_fn, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=CBMDataset.collate_fn, num_workers=8)

    args.orig_d_s = args.orig_d_nF = args.orig_d_fF = args.orig_d_acc = 24
    args.output_dim = len(train_ds.dataset.class2idx)

    model = MULT4Model(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        start = time.time()
        tr_loss, tr_acc, tr_f1 = train_epoch(model, optimizer, criterion, train_loader, device)
        va_loss, va_acc, va_f1 = eval_epoch(model, criterion, test_loader, device)
        scheduler.step(va_loss)
        elapsed = time.time() - start

        print(f"[Epoch {epoch+1:02d}/{args.epochs}] Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} F1: {va_f1:.4f} [{elapsed:.1f}s]")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            save_model(args, model, name=f'fold{args.fold}')
            #save_model(args, model, name=f'fold{args.fold}_fine')

    print("=== Final Evaluation on Test Fold ===")
    ckpt = torch.load(f"pre_trained_models/fold{args.fold}_{args.model}.pt")
    #ckpt = torch.load(f"pre_trained_models/fold{args.fold}_fine_{args.model}.pt")
    model.load_state_dict(ckpt)
    te_loss, te_acc, te_f1 = eval_epoch(model, criterion, test_loader, device)
    print(f"Test Loss {te_loss:.4f} TeAcc {te_acc:.4f} Test F1 {te_f1:.4f}")

    # device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')

    # train_ds_full = CBMDataset(args.data_path, add_noise=True, noise_snr=args.noise_snr,
    #                            use_cache=args.use_cache, train=True)
    # test_ds_full = CBMDataset(args.data_path, add_noise=False,
    #                           use_cache=args.use_cache, train=False)
    # train_ds, _ = train_ds_full.split_five_fold(args.fold)
    # _, test_ds = test_ds_full.split_five_fold(args.fold)

    # train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
    #                           collate_fn=CBMDataset.collate_fn, num_workers=8)
    # test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
    #                          collate_fn=CBMDataset.collate_fn, num_workers=8)

    # args.orig_d_s = args.orig_d_nF = args.orig_d_fF = args.orig_d_acc = 24
    # args.output_dim = len(train_ds.dataset.class2idx)

    # model = MULT4Model(args).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.CrossEntropyLoss()
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)

    # best_val_f1 = 0.0
    # for epoch in range(args.epochs):
    #     start = time.time()
    #     tr_loss, tr_acc, tr_f1 = train_epoch(model, optimizer, criterion, train_loader, device)
    #     va_loss, va_acc, va_f1 = eval_epoch(model, criterion, test_loader, device)
    #     scheduler.step(va_loss)
    #     elapsed = time.time() - start

    #     print(f"[Epoch {epoch+1:02d}/{args.epochs}] "
    #           f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} | "
    #           f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} F1: {va_f1:.4f} [{elapsed:.1f}s]")

    #     if va_f1 > best_val_f1:
    #         best_val_f1 = va_f1
    #         save_model(args, model, name=f'fold{args.fold}')

    # print("=== Final Evaluation on Test Fold ===")
    # ckpt = torch.load(f"pre_trained_models/fold{args.fold}_{args.model}.pt")
    # model.load_state_dict(ckpt)
    # te_loss, te_acc, te_f1 = eval_epoch(model, criterion, test_loader, device)
    # print(f"Test Loss {te_loss:.4f} Acc {te_acc:.4f} F1 {te_f1:.4f}")

if __name__ == '__main__':
    main()


    
# import os
# import time
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.data import DataLoader

# from src.utils import get_data, save_model, load_model
# from src.CBM_dataset import CBMDataset
# from src.models import MULT4Model
# import random

# from sklearn.metrics import accuracy_score, f1_score

# def train_epoch(model, optimizer, criterion, loader, device):
#     model.train()
#     total_loss = 0.0
#     all_preds, all_labels = [], []
#     for (idxs, snd, nF, fF, acc), ys, _ in loader:
#         snd, nF, fF, acc, ys = [t.to(device) for t in (snd, nF, fF, acc, ys)]
#         optimizer.zero_grad()
#         preds, _ = model(snd, nF, fF, acc)
#         loss = criterion(preds, ys)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * ys.size(0)
#         all_preds.append(preds.argmax(dim=1).cpu())
#         all_labels.append(ys.cpu())

#     avg_loss = total_loss / len(loader.dataset)
#     preds = torch.cat(all_preds)
#     labels = torch.cat(all_labels)
#     acc = accuracy_score(labels, preds)
#     f1 = f1_score(labels, preds, average='macro')
#     return avg_loss, acc, f1

# def eval_epoch(model, criterion, loader, device):
#     model.eval()
#     total_loss = 0.0
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for (idxs, snd, nF, fF, acc), ys, _ in loader:
#             snd, nF, fF, acc, ys = [t.to(device) for t in (snd, nF, fF, acc, ys)]
#             preds, _ = model(snd, nF, fF, acc)
#             loss = criterion(preds, ys)
#             total_loss += loss.item() * ys.size(0)
#             all_preds.append(preds.argmax(dim=1).cpu())
#             all_labels.append(ys.cpu())

#     avg_loss = total_loss / len(loader.dataset)
#     preds = torch.cat(all_preds)
#     labels = torch.cat(all_labels)
#     acc = accuracy_score(labels, preds)
#     f1 = f1_score(labels, preds, average='macro')
#     return avg_loss, acc, f1

# def main():
#     parser = argparse.ArgumentParser(description="CBM 4-modality MuIT Training")

#     # —— 数据 & 训练参数 —— 
#     parser.add_argument('--data_path', type=str,
#                         default='/home/manager/lkl/AAAI26_Multimodal/dataset/m1474014/CBM_FinalDatabase',
#                         help='CBM_FinalDatabase 根目录')
#     parser.add_argument('--fold', type=int, default=0, help='Five-fold Index 0-4')
#     parser.add_argument('--batch_size', type=int, default=64)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--lr', type=float, default=1e-3)
#     parser.add_argument('--use_cuda', action='store_true')
#     parser.add_argument('--add_noise', action='store_true')
#     parser.add_argument('--noise_snr', type=float, default=5.0)
#     parser.add_argument('--fill_missing', action='store_true')
#     parser.add_argument('--use_cache', action='store_true', help='是否使用pickle缓存数据')

#     parser.add_argument('--num_heads', type=int, default=4)
#     parser.add_argument('--layers', type=int, default=3)
#     parser.add_argument('--attn_dropout', type=float, default=0.1)
#     parser.add_argument('--relu_dropout', type=float, default=0.1)
#     parser.add_argument('--embed_dropout', type=float, default=0.25)
#     parser.add_argument('--res_dropout', type=float, default=0.1)
#     parser.add_argument('--out_dropout', type=float, default=0.1)
#     parser.add_argument('--attn_mask', action='store_true', help='是否使用attention mask')

#     parser.add_argument('--model', type=str, default='MULT4', help='模型名称')
#     parser.add_argument('--aligned', action='store_true', help='模型是否对齐')

#     args = parser.parse_args()

#     device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
#     torch.manual_seed(42)
#     random.seed(42)
#     if device.type == 'cuda':
#         torch.cuda.manual_seed(42)

#     # —— 加载数据 —— 
#     train_ds = get_data(args, split='train')
#     test_ds  = get_data(args, split='test')

#     # 设置随机加噪 (训练集启用，测试集关闭)
#     train_ds.dataset.random_noise_ratio = 0.2
#     train_ds.dataset.train = True

#     test_ds.dataset.random_noise_ratio = 0.0
#     test_ds.dataset.train = False

#     train_loader = DataLoader(
#         train_ds, batch_size=args.batch_size, shuffle=True,
#         collate_fn=CBMDataset.collate_fn, num_workers=8, pin_memory=True
#     )
#     test_loader = DataLoader(
#         test_ds, batch_size=args.batch_size, shuffle=False,
#         collate_fn=CBMDataset.collate_fn, num_workers=8, pin_memory=True
#     )

#     # —— 模型构造 —— 
#     args.orig_d_s = args.orig_d_nF = args.orig_d_fF = args.orig_d_acc = 24
#     args.output_dim = len(train_ds.dataset.class2idx)

#     model = MULT4Model(args).to(device)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     criterion = nn.CrossEntropyLoss()
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

#     best_val_f1 = 0.0

#     # —— 训练循环 —— 
#     for epoch in range(1, args.epochs + 1):
#         start = time.time()
#         tr_loss, tr_acc, tr_f1 = train_epoch(model, optimizer, criterion, train_loader, device)
#         va_loss, va_acc, va_f1 = eval_epoch(model, criterion, test_loader, device)
#         scheduler.step(va_loss)
#         elapsed = time.time() - start

#         print(f"[Epoch {epoch:02d} | {elapsed:.1f}s]  "
#               f"TrLoss {tr_loss:.4f}  TrAcc {tr_acc:.4f} TrF1 {tr_f1:.4f}  |  "
#               f"VaLoss {va_loss:.4f}  ValAcc {va_acc:.4f} VaF1 {va_f1:.4f}")

#         # 保存最佳模型
#         if va_f1 > best_val_f1:
#             best_val_f1 = va_f1
#             save_model(args, model, name=f"fold{args.fold}")

#     # —— 最终测试 —— 
#     print("=== Final Evaluation on Test Fold ===")
#     model = load_model(args, MULT4Model, name=f"fold{args.fold}")
#     model.to(device)
#     te_loss, te_acc, te_f1 = eval_epoch(model, criterion, test_loader, device)
#     print(f"Test Loss {te_loss:.4f} TeAcc {te_acc:.4f} Test F1 {te_f1:.4f}")

# if __name__ == '__main__':
#     main()
