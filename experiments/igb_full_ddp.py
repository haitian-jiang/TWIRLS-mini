import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import torch.multiprocessing as mp
import time, tqdm, numpy as np
from igb.dataloader import IGB260MDGLDataset
from torch_geometric.seed import seed_everything
from queue import Queue
import threading
import torch.distributed as dist
import sys

sys.path.append("../")
from model import GNNModel, APPNP


def train(model, graph, x, y, loss_func, optimizer):
    output = model(graph, x)[: y.shape[0]]
    loss = loss_func(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, graph, x, y):
    output = model(graph, x)[: y.shape[0]]
    correct = (output.argmax(-1) == y).sum().item()
    total = y.size(0)
    return correct, total


def graph_load(args, queue, sequence, fanout_info, g: dgl.DGLGraph, device):
    for i in sequence:
        graph: dgl.DGLGraph = torch.load(f"{args.dir}/{args.dataset}-{fanout_info}/train-{i}.pt")
        x = g.ndata["feat"][graph.ndata["_ID"]].float().to(device, non_blocking=True)
        y = g.ndata["label"][graph.ndata["_ID"][graph.split_idx]].long().to(device, non_blocking=True)
        graph = graph.to(device, non_blocking=True)
        queue.put((graph, x, y))


def run(proc_id, devices, g, args, sequence):
    dev_id = devices[proc_id]
    dist_init_method = "tcp://{master_ip}:{master_port}".format(master_ip="127.0.0.1", master_port="12345")
    if torch.cuda.device_count() < 1:
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo", init_method=dist_init_method, world_size=len(devices), rank=proc_id)
    else:
        torch.cuda.set_device(dev_id)
        device = torch.device("cuda:" + str(dev_id))
        dist.init_process_group(backend="nccl", init_method=dist_init_method, world_size=len(devices), rank=proc_id)

    sampler = dgl.dataloading.ShaDowKHopSampler(eval(args.fanout))

    # train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    num_total_train_batches = 95102
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]

    val_dataloader = dgl.dataloading.NodeDataLoader(
        g, val_nid, sampler, device="cpu", use_ddp=True, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0
    )

    test_dataloader = dgl.dataloading.NodeDataLoader(
        g, test_nid, sampler, device="cpu", use_ddp=True, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0
    )

    in_feats = g.ndata["feat"].shape[1]

    model = GNNModel(
        in_feats,
        args.num_classes,
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

    if device == torch.device("cpu"):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None, output_device=None)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)

    loss_fcn = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay)

    fanout_info = args.fanout[1:-1].replace(",", "-")

    # Training loop
    for epoch in range(args.epochs):
        # Loop over the dataloader to sample the computation dependency graph as a list of blocks.

        graph_queue = Queue(maxsize=5)
        loader = threading.Thread(target=graph_load, args=(args, graph_queue, sequence, fanout_info, g, device), daemon=True)
        loader.start()

        epoch_loss = 0
        model.train()
        for i in tqdm(sequence):
            graph, x, y = graph_queue.get()
            loss = train(model, graph, x, y, loss_fcn, optimizer)
            epoch_loss += loss.detach()
        print(f"Epoch: {epoch}\t" f"Loss: {epoch_loss:.10f}")

        model.eval()
        valid_correct, valid_tot, val_acc = 0, 0, 0
        for i, (input_nodes, output_nodes, graph) in enumerate(tqdm(val_dataloader)):
            x = graph.ndata["feat"].float().to(device)
            y = graph.ndata["label"][output_nodes].long().to(device)
            graph = graph.to(device)
            correct, tot = evaluate(model, graph, x, y)
            valid_correct += correct
            valid_tot += tot
        val_acc = valid_correct / valid_tot
        val_acc = torch.tensor(val_acc)
        dist.reduce(val_acc, 0)
        # --- test --- #
        test_correct, test_tot, test_acc = 0, 0, 0
        for i, (input_nodes, output_nodes, graph) in enumerate(tqdm(test_dataloader)):
            x = g.ndata["feat"].float().to(device)
            y = g.ndata["label"][output_nodes].long().to(device)
            graph = graph.to(device)
            correct, tot = evaluate(model, graph, x, y)
            test_correct += correct
            test_tot += tot
        test_acc = test_correct / test_tot
        test_acc = torch.tensor(test_acc)
        dist.reduce(test_acc, 0)
        if val_acc.item() > best_val_acc:
            best_val_acc = val_acc.item()
            best_test_acc = test_acc.item()
            best_epoch = epoch
        if proc_id == 0:
            print(f"Epoch: {epoch}\t" f"Valid acc: {val_acc.item() * 100:.2f}%\t" f"Test acc: {test_acc.item() * 100:.2f}%\t")
    print(f"Best epoch: {best_epoch}")
    print(f"Valid acc: {best_val_acc * 100:.2f}%")
    print(f"Test acc: {best_test_acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument("--path", type=str, default="/mnt/nvme14/IGB260M/", help="path containing the datasets")
    parser.add_argument("--dataset_size", type=str, default="tiny", choices=["tiny", "small", "medium", "large", "full"], help="size of the datasets")
    parser.add_argument("--num_classes", type=int, default=19, choices=[19, 2983], help="number of classes")
    parser.add_argument("--in_memory", type=int, default=1, choices=[0, 1], help="0:read only mmap_mode=r, 1:load into memory")
    parser.add_argument("--synthetic", type=int, default=0, choices=[0, 1], help="0:nlp-node embeddings, 1:random")
    parser.add_argument("--dir", type=str, default="/nvme1n1/offline_subgraph")

    # Model
    parser.add_argument("--epochs", type=int, default=10)
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

    # Model parameters
    parser.add_argument("--fanout", type=str, default="[5,10,15]")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--gpu_devices", type=str, default="0,1,2,3")
    args = parser.parse_args()
    seed_everything(args.seed)

    gpu_idx = [int(fanout) for fanout in args.gpu_devices.split(",")]
    num_gpus = len(gpu_idx)
    dataset = IGB260MDGLDataset(args)
    g = dataset[0]
    print(g)

    mp.spawn(
        run,
        args=(
            gpu_idx,
            g,
            args,
        ),
        nprocs=num_gpus,
    )
