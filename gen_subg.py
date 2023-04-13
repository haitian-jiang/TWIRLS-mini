import random
import argparse
import torch
from torch_geometric.seed import seed_everything
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import trange
import dgl


def index2mask(idx, size: int):
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

def preprocess_ogb_data(dataset, args):
    graph, label = dataset[0]
    g = dgl.to_bidirected(graph, copy_ndata=True)
    label = label.view(-1)
    split_idx = dataset.get_idx_split()
    g.split_idx = split_idx
    if args.dataset == "papers100M":
        label = label.view(-1)
        label[label.isnan()]=404.
        label = label.long()
    g.ndata['label'] = label
    return g


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', type=str, default='arxiv')
    parser.add_argument('--sampler', type=str, default='shadowkhop')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--khop', type=int, default=2)
    args = parser.parse_args()
    return args


def gen_fullkhop_subg(graph, args, new_dataset):
    for split, idx in graph.split_idx.items():
            ilist = idx.view(-1).tolist()
            random.shuffle(ilist)
            print(f"Start {split}")

            for begin in trange(0, len(ilist), args.batch_size):
                subset = ilist[begin: begin + args.batch_size]
                subset.sort()

                # ndata={'feat':, 'label':, '_ID':}, subg={ndata, split_idx, split={'train','valid','test'}}
                subg, idx = dgl.khop_in_subgraph(graph, subset, args.khop, relabel_nodes=True)
                subg.split_idx = idx
                subg.split = split
                new_dataset[split].append(subg)
    return new_dataset


def gen_shadowkhop_subg(graph, args, new_dataset):
    sampler = dgl.dataloading.ShaDowKHopSampler([5, 10, 15])
    for split, idx in graph.split_idx.items():
        dataloader = dgl.dataloading.DataLoader(graph, idx, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
        for input_nodes, output_nodes, subgraph in dataloader:
            idx = torch.arange(len(output_nodes))
            subgraph.split_idx = idx
            subgraph.split = split
            new_dataset[split].append(subgraph)
    return new_dataset

if __name__ == '__main__':
    args = parse_arg()
    seed_everything(args.seed)

    # if args.dataset.lower() in ["arxiv", "products", "papers100m"]:
    dataset = DglNodePropPredDataset(name=f'ogbn-{args.dataset}', root='/home/ubuntu/dataset')
    graph = preprocess_ogb_data(dataset, args)

    args.num_classes = dataset.num_classes
    args.num_features = graph.ndata['feat'].shape[-1]

    print("Dataset ready")

    new_dataset = {split: [] for split in ['train', 'valid', 'test']}
    new_dataset['num_classes'] = args.num_classes
    new_dataset['num_features'] = args.num_features

    if args.sampler == 'fullkhop':
        new_dataset = gen_fullkhop_subg(graph, args, new_dataset)
    elif args.sampler == 'shadowkhop':
        new_dataset = gen_shadowkhop_subg(graph, args, new_dataset)
    
    torch.save(new_dataset, f'./dataset/{args.dataset}-{args.sampler}.pt')
