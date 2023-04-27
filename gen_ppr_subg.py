import sys
sys.path.append('/home/ubuntu/ibmb')
from dataloaders import IBMBNodeLoader
import random
import argparse
import dgl
import torch
from torch_geometric.seed import seed_everything
from ogb.nodeproppred import PygNodePropPredDataset
from tqdm import trange, tqdm
from torch_geometric.utils import to_undirected

def to_dgl(data):
    row, col, _ = data.edge_index.coo()
    g = dgl.graph((row, col))
    for attr in data.node_attrs():
        if attr == "x":
            g.ndata['feat'] = data[attr]
        elif attr == "y":
            g.ndata['label'] = data[attr]
        else:
            g.ndata[attr] = data[attr]
    for attr in data.edge_attrs():
        if attr in ['edge_index', 'adj_t']:
            continue
        g.edata[attr] = data[attr]
    g.split_idx = data.output_node_mask
    return g



def index2mask(idx, size: int):
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

def preprocess_ogb_data(dataset, args):
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index)
    data.y = data.y.view(-1)
    split_idx = dataset.get_idx_split()
    data.split_idx = split_idx
    if args.dataset == "papers100M":
        data.y[data.y.isnan()] = 404.
        data.y = data.y.long()
    return data


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', type=str, default='arxiv')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--topk', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--eps', type=float, default=2e-4)
    args = parser.parse_args()
    return args

def gen_ppr_subg(graph, args, new_dataset):
    # train_ld = IBMBNodeLoader(graph, 'order', graph.split_idx['train'], 'adj', args.topk, args.batch_size, None, args.alpha, args.eps, shuffle=False, batch_size=1)
    # valid_ld = IBMBNodeLoader(graph, 'order', graph.split_idx['valid'], 'adj', args.topk, args.batch_size, None, args.alpha, args.eps, shuffle=False, batch_size=1)
    # test_ld = IBMBNodeLoader(graph, 'order', graph.split_idx['test'], 'adj', args.topk, args.batch_size, None, args.alpha, args.eps, shuffle=False, batch_size=1)
    train_ld = IBMBNodeLoader(graph, 'order', graph.split_idx['train'], 'adj', args.topk, args.batch_size, None, args.alpha, args.eps, shuffle=False, batch_size=1)
    valid_ld = IBMBNodeLoader(graph, 'order', graph.split_idx['valid'], 'adj', args.topk, args.batch_size, None, args.alpha, args.eps, shuffle=False, batch_size=1)
    test_ld = IBMBNodeLoader(graph, 'order', graph.split_idx['test'], 'adj', args.topk, args.batch_size, None, args.alpha, args.eps, shuffle=False, batch_size=1)
    for split, loader in zip(['train', 'valid', 'test'], [train_ld, valid_ld, test_ld]):
        new_dataset[split] = [to_dgl(subg) for subg in loader]
    return new_dataset

if __name__ == '__main__':
    args = parse_arg()
    seed_everything(args.seed)

    # if args.dataset.lower() in ["arxiv", "products", "papers100m"]:
    dataset = PygNodePropPredDataset(name=f'ogbn-{args.dataset}', root='/home/ubuntu/dataset')
    graph = preprocess_ogb_data(dataset, args)

    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_features

    print("Dataset ready")

    new_dataset = {split: [] for split in ['train', 'valid', 'test']}
    new_dataset['num_classes'] = args.num_classes
    new_dataset['num_features'] = args.num_features

    new_dataset = gen_ppr_subg(graph, args, new_dataset)
    
    torch.save(new_dataset, f'./dataset/{args.dataset}-ppr.pt')
