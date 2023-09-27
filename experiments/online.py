import random
import torch
from tqdm import tqdm, trange
import sys

sys.path.append("../")
from model import GNNModel, APPNP
import dgl
import dgl.nn as dglnn
from dgl.utils import gather_pinned_tensor_rows
import torch.nn as nn
import torch.nn.functional as F
import pdb
import argparse
from torch_geometric.seed import seed_everything
from ogb.nodeproppred import DglNodePropPredDataset
import time
import numpy as np
import os
import psutil
from load_graph import *


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


def train(model, graph, x, y, loss_func, optimizer):
    model.train()
    output = model(graph, x)[: y.shape[0]]
    loss = loss_func(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, graph):
    model = model.eval()
    nodes = graph.split_idx
    output = model(graph, graph.ndata["feat"])[nodes]
    labels = graph.ndata["label"][nodes]
    correct = (output.argmax(-1) == labels).sum().item()
    total = labels.size(0)
    return correct, total


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch GNN")
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--fanout", type=str, default="[5,10,15]")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--model", type=str, default="twirls")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--pre_mlp", type=int, default=0)
    parser.add_argument("--aft_mlp", type=int, default=3)
    parser.add_argument("--precond", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--lmbd", type=float, default=1)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=5e-4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--skip", type=int, default=1)
    parser.add_argument("--inp_dropout", type=float, default=0.2)
    parser.add_argument("--rec_energy", type=int, default=0)
    parser.add_argument("--path", type=str, default="/efs/rjliu/dataset/igb_dataset")
    parser.add_argument("--dataset_size", type=str, default="tiny", choices=["tiny", "small", "medium", "large", "full"])
    parser.add_argument("--in_memory", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--synthetic", type=int, default=0)
    parser.add_argument('--activate', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print(args)
    seed_everything(args.seed)
    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    to_bidirected, label_offset = False, 0
    if args.dataset.startswith("ogbn"):
        if args.dataset == "ogbn-products":
            to_bidirected = True
        dataset = load_ogb(args.dataset, "/efs/rjliu/dataset", to_bidirected)
    elif args.dataset.startswith("igb"):
        if args.dataset == "igb-small":
            to_bidirected = True
        dataset = load_igb(args, to_bidirected)
    elif args.dataset == "mag240m":
        dataset = load_mag240m("/home/ubuntu/mag", only_graph=False)
        label_offset = dataset[-1]
        dataset = dataset[:-1]
    else:
        raise NotImplementedError

    print(dataset[0])
    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Graph total memory:", mem1 - mem, "GB")

    g, feats, labels, n_classes, splitted_idx = dataset
    output_channels = n_classes

    input_channels = feats.shape[1]
    print("Feature dimension:", input_channels)
    device = torch.device(f"cuda:{args.device}")

    # feats = feats.pin_memory()

    # --- init model --- #
    if args.model == "twirls":
        model = GNNModel(
            input_channels,
            output_channels,
            args.hidden,
            args.K,
            args.pre_mlp,
            args.aft_mlp,
            "none",
            args.precond,
            args.alpha,
            args.lmbd,
            dropout=args.dropout,
            skip=args.skip,
            inp_dropout=args.inp_dropout,
            rec_energy=args.rec_energy,
            activate=args.activate,
        )
    # model = SAGE(input_channels, 256, output_channels)
    elif args.model == "APPNP":
        model = APPNP(input_channels, [args.hidden] * 2, output_channels, F.relu, args.dropout, 0, args.alpha, args.K)
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    # def prepare_graph(g):
    #     g = g.remove_self_loop()
    #     g = g.add_self_loop()
    #     return g

    # dataset['train'] = [prepare_graph(g) for g in dataset['train']]
    # dataset['valid'] = [prepare_graph(g) for g in dataset['valid']]
    # dataset['test'] = [prepare_graph(g) for g in dataset['test']]

    best_val_acc, best_test_acc, best_epoch = 0, 0, 0

    sampler = dgl.dataloading.ShaDowKHopSampler(eval(args.fanout))
    # sampler = dgl.dataloading.NeighborSampler(eval(args.fanout))
    train_nid = splitted_idx["train"][: 95102 * args.batch_size] if args.dataset == "igb-full" else splitted_idx["train"]
    if args.dataset in ("igb-full", "igb-large", "igb-medium", "mag240m"):
        perm_idx = torch.randperm(train_nid.numel())[:train_nid.numel() // 10]
        train_nid = train_nid[perm_idx]
    dataloader = dgl.dataloading.DataLoader(g, train_nid, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    epoch_time_list = []
    sample_time_list = []
    feat_load_list = []
    graph_load_list = []
    train_time_list = []

    for e in range(args.epochs):
        # --- train --- #
        tot_loss = 0
        sampling_time, feat_load_time, graph_load_time, training_time = 0, 0, 0, 0
        model.train()
        torch.cuda.synchronize()
        start = time.time()
        tic = time.time()
        for i, (input_nodes, output_nodes, graph) in enumerate(tqdm(dataloader)):
            # torch.cuda.synchronize()
            # sampling_time += time.time() - tic

            # tic = time.time()
            # x = gather_pinned_tensor_rows(feats, input_nodes.to(device)).float()
            x = feats[input_nodes].float().to(device)
            y = labels[output_nodes - label_offset].long().to(device)
            # torch.cuda.synchronize()
            # feat_load_time += time.time() - tic

            # tic = time.time()
            graph = graph.to(device)
            # torch.cuda.synchronize()
            # graph_load_time += time.time() - tic

            # tic = time.time()
            loss = train(model, graph, x, y, loss_func, optimizer)
            tot_loss += loss
            # torch.cuda.synchronize()
            # training_time += time.time() - tic

            # torch.cuda.synchronize()
            # tic = time.time()
        torch.cuda.synchronize()
        epoch_time = time.time() - start
        epoch_time_list.append(epoch_time)
        sample_time_list.append(sampling_time)
        feat_load_list.append(feat_load_time)
        graph_load_list.append(graph_load_time)
        train_time_list.append(training_time)
        print(
            f"Epoch Time: {epoch_time:.3f}\t"
            f"Sample Time: {sampling_time:.3f}\t"
            f"Feat Load Time: {feat_load_time:.3f}\t"
            f"Graph Load Time: {graph_load_time:.3f}\t"
            f"Train Time: {training_time:.3f}"
        )
    print(f"Best epoch: {best_epoch}")
    print(f"Valid acc: {best_val_acc * 100:.2f}%")
    print(f"Test acc: {best_test_acc * 100:.2f}%")
    print(f"Avg Epoch Time: {np.mean(epoch_time_list[1:]):.3f}s")
    print(f"Avg Sample Time: {np.mean(sample_time_list[1:]):.3f}s")
    print(f"Avg Feat Load Time: {np.mean(feat_load_list[1:]):.3f}s")
    print(f"Avg Graph Load Time: {np.mean(graph_load_list[1:]):.3f}s")
    print(f"Avg Train Time: {np.mean(train_time_list[1:]):.3f}s")


if __name__ == "__main__":
    main()
