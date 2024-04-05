import argparse
import dgl
import torch
from torch_geometric.seed import seed_everything
from tqdm import trange, tqdm
from load_graph import *
import psutil
import numba
from numba.core import types
from numba.typed import Dict
import numpy as np
from ShadowLabor import ShaDowLabor


@numba.njit
def find_indices_in(a, b):
    d = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i, v in enumerate(b):
        d[v] = i
    ai = np.zeros_like(a)
    for i, v in enumerate(a):
        ai[i] = d.get(v, -1)
    return ai


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--dir", type=str, default="/opt/dlami/nvme/offline")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--fanout", type=str, default="[5,10,15]")
    parser.add_argument("--path", type=str, default="/home/ubuntu/dataset/igb_dataset")
    parser.add_argument("--dataset_size", type=str, default="tiny", choices=["tiny", "small", "medium", "large", "full"])
    parser.add_argument("--in_memory", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--synthetic", type=int, default=0)
    parser.add_argument("--sampler", type=str, default="shadowkhop", choices=["shadowkhop", "shadowlabor", "neighbor_sampler"])
    args = parser.parse_args()
    return args


def gen_shadowkhop_subg(graph, split_idx, args):
    sampler = dgl.dataloading.ShaDowKHopSampler(eval(args.fanout))
    fanout_info = args.fanout[1:-1].replace(",", "-")
    output_dir = f"{args.dir}/{args.dataset}-{fanout_info}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for split, idx in split_idx.items():
        if split == "train":
            dataloader = dgl.dataloading.DataLoader(graph, idx, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
            for i, (input_nodes, output_nodes, subgraph) in enumerate(tqdm(dataloader)):
                subgraph.split_idx = torch.arange(len(output_nodes))
                subgraph = subgraph.remove_self_loop()
                subgraph = subgraph.add_self_loop()
                torch.save(subgraph, f"{output_dir}/{split}-{i}.pt")


def gen_shadowlabor_subg(graph, split_idx, args):
    sampler = dgl.dataloading.LaborSampler(eval(args.fanout))
    fanout_info = args.fanout[1:-1].replace(",", "-")
    output_dir = f"{args.dir}/{args.dataset}-{fanout_info}-lb"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for split, idx in split_idx.items():
        if split == "train":
            dataloader = dgl.dataloading.DataLoader(graph, idx, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
            for i, (input_nodes, output_nodes, blk) in enumerate(tqdm(dataloader)):
                subgraph = graph.subgraph(input_nodes, relabel_nodes=True)
                assert (input_nodes[:len(output_nodes)] == output_nodes).all()
            # for i, (input_nodes, output_nodes, subgraph) in enumerate(tqdm(dataloader)):
                subgraph.split_idx = torch.arange(len(output_nodes))
                subgraph = subgraph.remove_self_loop()
                subgraph = subgraph.add_self_loop()
                torch.save(subgraph, f"{output_dir}/{split}-{i}.pt")



def gen_neighbor_sampler_subg(graph: dgl.DGLGraph, split_idx, args):
    sampler = dgl.dataloading.NeighborSampler(eval(args.fanout))
    fanout_info = args.fanout[1:-1].replace(",", "-")
    output_dir = f"{args.dir}/{args.dataset}-{fanout_info}-ns"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    coo_g = graph.formats("coo")
    for split, idx in split_idx.items():
        if split == "train":
            dataloader = dgl.dataloading.DataLoader(graph, idx, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False)
            for i, (input_nodes, output_nodes, blocks) in enumerate(tqdm(dataloader)):
                eids = torch.cat([block.edata[dgl.EID] for block in reversed(blocks)])
                subgraph = dgl.edge_subgraph(coo_g, eids, relabel_nodes=True)
                rev_idx = find_indices_in(output_nodes.numpy(), subgraph.ndata[dgl.NID].numpy())
                subgraph.split_idx = torch.from_numpy(rev_idx)
                subgraph = subgraph.remove_self_loop()
                subgraph = subgraph.add_self_loop()
                torch.save(subgraph, f"{output_dir}/{split}-{i}.pt")


if __name__ == "__main__":
    args = parse_arg()
    print(args)
    seed_everything(args.seed)

    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024 * 1024)

    to_bidirected = False
    if args.dataset.startswith("ogbn"):
        if args.dataset == "ogbn-products":
            to_bidirected = True
        dataset = load_ogb(args.dataset, "/opt/dlami/nvme/dataset/OGB", to_bidirected)
    elif args.dataset.startswith("igb"):
        if args.dataset == "igb-small":
            to_bidirected = True
        dataset = load_igb(args, to_bidirected)
    elif args.dataset == "mag240m":
        dataset = load_mag240m("/home/ubuntu/mag", only_graph=True)
    else:
        raise NotImplementedError
    print(dataset[0])
    mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
    print("Graph total memory:", mem1 - mem, "GB")

    print("Dataset ready")
    if args.sampler == "shadowkhop":
        gen_shadowkhop_subg(dataset[0], dataset[4], args)
    elif args.sampler == "neighbor_sampler":
        gen_neighbor_sampler_subg(dataset[0], dataset[4], args)
    elif args.sampler == "shadowlabor":
        gen_shadowlabor_subg(dataset[0], dataset[4], args)
