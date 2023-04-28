import sys
sys.path.append('../')
from model import MLP
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pdb
import argparse
from torch_geometric.seed import seed_everything
from torch.utils.data import DataLoader
from math import ceil


@torch.no_grad()
def evaluate(model, feature, idx, label):
    model = model.eval()
    output = model(feature[idx])
    correct = (output.argmax(-1) == label[idx]).sum().item()
    total = idx.size(0)
    return correct, total


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GNN')
    parser.add_argument('--dataset', type=str, default='7-1-1')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--log_every', type=int, default=-1)
    parser.add_argument('--early_stop', type=int, default=100)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_num', type=int, default=1)
    parser.add_argument('--skip', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    seed_everything(args.seed)
    input_channels = 128
    output_channels = 172
    device = torch.device(f'cuda:{args.device}')
    # --- load data --- #
    feature = np.fromfile(f'/home/ubuntu/TWIRLS/dataset/100M/{args.dataset}.bin', dtype=np.float32).reshape(-1, input_channels)
    feature = torch.from_numpy(feature)
    label = np.load('/home/ubuntu/dataset/ogbn_papers100M/raw/node-label.npz')['node_label']
    label = torch.from_numpy(label).view(-1)
    label = label[~label.isnan()].long().to(device)
    split_idx = torch.load('/home/ubuntu/TWIRLS/dataset/100M/compressed_split_idx.pt')
    feature = feature.to(device)

    # --- init model --- #
    model = MLP(input_channels, args.hidden, output_channels, args.layers, args.dropout, 'batch', False, skip=args.skip)
    # model = SAGE(input_channels, 256, output_channels)
    # model = APPNP(input_channels, [512, 512], output_channels, F.relu, 0.5, 0, 0.05, 10)
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params          = model.parameters() , 
        lr              = args.lr , 
        weight_decay    = args.wd , 
    )

    best_val, best_test, best_epoch = 0, 0, 0
    for e in range(args.epochs):
        # --- train --- #
        if args.batch_num == 1:
            model.train()
            train_idx = split_idx['train']
            output = model(feature[train_idx])
            loss = loss_func(output, label[train_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.log_every > 0 and e % args.log_every == 0:
                print(f"Epoch {e:02d}: loss: {loss.item():.4f}", end='\t')
        else:
            model.train()
            train_idx = split_idx['train']
            train_loader = DataLoader(train_idx, batch_size=ceil(train_idx.size(0)/args.batch_num), shuffle=True)
            for batch_idx in train_loader:
                output = model(feature[batch_idx])
                loss = loss_func(output, label[batch_idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if args.log_every > 0 and e % args.log_every == 0:
                print(f"Epoch {e:02d}: loss: {loss.item():.4f}", end='\t')
        # --- valid ---#
        valid_idx = split_idx['valid']
        correct, tot = evaluate(model, feature, valid_idx, label)
        val_acc = correct / tot
        if args.log_every > 0 and e % args.log_every == 0:
            print(f"Valid acc: {val_acc * 100:.2f}%", end='\t')
        # --- test --- #
        test_idx = split_idx['test']
        correct, tot = evaluate(model, feature, test_idx, label)
        test_acc = correct / tot
        if args.log_every > 0 and e % args.log_every == 0:
            print(f"Test acc: {test_acc * 100:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_epoch = e
        # if e - best_epoch > args.early_stop:
        #     break
    K, alpha, lmbd = args.dataset.split('-')
    print(f"Valid acc: {best_val * 100:.2f}%,   Test acc: {best_test * 100:.2f}%      Epoch {best_epoch}\n"
    f"`{args.hidden}, K={K}, MLP=[0,3], alpha={alpha}, lmbd={lmbd}, dropout={args.dropout}, lr={args.lr}, wd={args.wd}`\n")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
