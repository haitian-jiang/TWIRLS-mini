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
from load_graph import *
import psutil
import numpy as np
from queue import Queue
import threading
import multiprocessing as mp


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
def evaluate(model, graph, x, y):
    model = model.eval()
    output = model(graph, x)[: y.shape[0]]
    correct = (output.argmax(-1) == y).sum().item()
    total = y.size(0)
    return correct, total


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch GNN")
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--dir", type=str, default="/nvme1n1/offline_subgraph")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--fanout", type=str, default="[5,10,15]")
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
    parser.add_argument("--path", type=str, default="/home/ubuntu/dataset/igb_dataset")
    parser.add_argument("--dataset_size", type=str, default="tiny", choices=["tiny", "small", "medium", "large", "full"])
    parser.add_argument("--in_memory", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--synthetic", type=int, default=0)
    parser.add_argument("--activate", type=int, default=0)
    args = parser.parse_args()
    return args


def graph_load(args, queue, sequence, fanout_info, feats: torch.Tensor, labels: torch.Tensor, label_offset, device):
    for i in sequence:
        graph: dgl.DGLGraph = torch.load(f"{args.dir}/{args.dataset}-{fanout_info}/train-{i}.pt")
        x = feats[graph.ndata["_ID"]].float().to(device, non_blocking=True)
        y = labels[graph.ndata["_ID"][graph.split_idx] - label_offset].long().to(device, non_blocking=True)
        graph = graph.to(device, non_blocking=True)
        queue.put((graph, x, y))


def main():
    args = parse_args()
    print(args)
    seed_everything(args.seed)
    fanout_info = args.fanout[1:-1].replace(",", "-")
    # --- load data --- #
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    to_bidirected, label_offset = False, 0
    if args.dataset.startswith("ogbn"):
        if args.dataset == "ogbn-products":
            to_bidirected = True
        dataset = load_ogb(args.dataset, "/home/ubuntu/dataset", to_bidirected)
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
    elif args.model == "APPNP":
        model = APPNP(input_channels, [args.hidden] * 2, output_channels, F.relu, args.dropout, 0, args.alpha, args.K)
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.wd,
    )

    sampler = dgl.dataloading.ShaDowKHopSampler(eval(args.fanout))
    num_total_batches = 95102 if args.dataset == "igb-full" else (splitted_idx["train"].numel() + args.batch_size - 1) // args.batch_size
    valid_loader = dgl.dataloading.DataLoader(
        g,
        splitted_idx["valid"],
        sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True,
        device=device,
    )
    test_loader = dgl.dataloading.DataLoader(
        g,
        splitted_idx["test"],
        sampler,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        use_uva=True,
        device=device,
    )

    # if args.dataset in ("igb-full", "igb-large", "igb-medium", "mag240m"):
    #     sequence = np.random.choice(num_total_batches, num_total_batches // 10, replace=False)
    # else:
    sequence = range(num_total_batches)

    best_val_acc, best_test_acc, best_epoch = 0, 0, 0
    epoch_time_list, train_time_list = [], []

    for e in range(args.epochs):
        # --- train --- #
        # num_nodes_list, num_edges_list = [], []
        tot_loss = 0
        train_time = 0
        # torch.cuda.synchronize()
        start = time.time()

        graph_queue = Queue(maxsize=5)
        loader = threading.Thread(target=graph_load, args=(args, graph_queue, sequence, fanout_info, feats, labels, label_offset, device), daemon=True)
        loader.start()

        for i in tqdm(sequence):
            graph, x, y = graph_queue.get()
            # graph = torch.load(f"{args.dir}/{args.dataset}-{fanout_info}-ns/train-{i}.pt")
            # x = gather_pinned_tensor_rows(feats, graph.ndata["_ID"].to(device)).float()

            # torch.cuda.synchronize()
            # tic = time.time()
            loss = train(model, graph, x, y, loss_func, optimizer)
            # torch.cuda.synchronize()
            # train_time += time.time() - tic

            tot_loss += loss

        loader.join()
        # torch.cuda.synchronize()
        epoch_time = time.time() - start
        epoch_time_list.append(epoch_time)
        train_time_list.append(train_time)
        # --- valid ---#
        valid_correct, valid_tot, val_acc = 0, 0, 0
        for i, (input_nodes, output_nodes, graph) in enumerate(tqdm(valid_loader)):
            # x = gather_pinned_tensor_rows(feats, input_nodes.to(device)).float()
            x = feats[input_nodes].float().to(device)
            y = labels[output_nodes - label_offset].long().to(device)
            graph = graph.to(device)
            correct, tot = evaluate(model, graph, x, y)
            valid_correct += correct
            valid_tot += tot
        val_acc = valid_correct / valid_tot
        # --- test --- #
        test_correct, test_tot, test_acc = 0, 0, 0
        for i, (input_nodes, output_nodes, graph) in enumerate(tqdm(test_loader)):
            # x = gather_pinned_tensor_rows(feats, input_nodes.to(device)).float()
            x = feats[input_nodes].float().to(device)
            y = labels[output_nodes - label_offset].long().to(device)
            graph = graph.to(device)
            correct, tot = evaluate(model, graph, x, y)
            test_correct += correct
            test_tot += tot
        test_acc = test_correct / test_tot
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = e
        if args.log_every > 0 and e % args.log_every == 0:
            print(
                f"Epoch: {e}\t"
                f"Loss: {tot_loss:.10f}\t"
                f"Valid acc: {val_acc * 100:.2f}%\t"
                f"Test acc: {test_acc * 100:.2f}%\t"
                f"Epoch Time: {epoch_time:.3f}s\t"
                f"Train Time: {train_time:.3f}s\t"
                # f"Avg #nodes: {np.mean(num_nodes_list):.3f}\t"
                # f"Avg #edges: {np.mean(num_edges_list):.3f}"
            )
    print(f"Best epoch: {best_epoch}")
    print(f"Valid acc: {best_val_acc * 100:.2f}%")
    print(f"Test acc: {best_test_acc * 100:.2f}%")
    print(f"Avg Epoch Time: {np.mean(epoch_time_list[1:]):.3f}s")
    if args.rec_energy:
        torch.save(model.unfolding.energy, f"./energy-{args.dataset}.pt")
        breakpoint()


if __name__ == "__main__":
    main()
