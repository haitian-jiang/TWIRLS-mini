import random
import argparse
import dgl
import torch
from torch_geometric.seed import seed_everything
from ogb.nodeproppred import DglNodePropPredDataset
from igb.dataloader import IGB260MDGLDataset
from tqdm import trange, tqdm
import pickle


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

def preprocess_igb_data(dataset, args):
    graph = dataset[0]
    g = dgl.to_bidirected(graph, copy_ndata=True)
    split_idx = {}
    split_idx['train'] = torch.where(g.ndata['train_mask'])[0]
    split_idx['valid'] = torch.where(g.ndata['val_mask'])[0]
    split_idx['test'] = torch.where(g.ndata['test_mask'])[0]
    g.split_idx = split_idx
    return g



def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--dataset', type=str, default='arxiv')
    parser.add_argument('--sampler', type=str, default='shadowkhop')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--khop', type=int, default=2)
    parser.add_argument('--fanout', type=str, default='[5,10,15]')
    parser.add_argument('--gb', type=str, default='ogb')
    parser.add_argument('--path', type=str, default='/home/ubuntu/dataset/igb')
    parser.add_argument('--dataset_size', type=str, default='tiny')
    parser.add_argument('--in_memory', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--synthetic', type=int, default=0)
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
    # sampler = dgl.dataloading.NeighborSampler(eval(args.fanout))
    sampler = dgl.dataloading.ShaDowKHopSampler(eval(args.fanout))
    for split, idx in graph.split_idx.items():
        dataloader = dgl.dataloading.DataLoader(graph, idx, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
        for i, (input_nodes, output_nodes, subgraph) in enumerate(tqdm(dataloader)):
            # breakpoint()
            idx = torch.arange(len(output_nodes))
            subgraph.split_idx = idx
            subgraph.split = split
            subgraph = subgraph.remove_self_loop()
            subgraph = subgraph.add_self_loop()
            # new_dataset[split].append(subgraph)
            torch.save(subgraph, f'../dataset/{args.dataset}-10-15/{split}-{i}.pt')
    return new_dataset

if __name__ == '__main__':
    args = parse_arg()
    seed_everything(args.seed)

    # if args.dataset.lower() in ["arxiv", "products", "papers100m"]:
    if args.gb == 'ogb':
        dataset = DglNodePropPredDataset(name=f'ogbn-{args.dataset}', root='/home/ubuntu/dataset')
        graph = preprocess_ogb_data(dataset, args)
        args.num_classes = dataset.num_classes
    elif args.gb == 'igb':
        dataset = IGB260MDGLDataset(args)
        graph = preprocess_igb_data(dataset, args)
    
    # args.num_features = graph.ndata['feat'].shape[-1]
    args.num_features = 100

    print("Dataset ready")

    new_dataset = {split: [] for split in ['train', 'valid', 'test']}
    new_dataset['num_classes'] = args.num_classes
    new_dataset['num_features'] = args.num_features

    if args.sampler == 'fullkhop':
        new_dataset = gen_fullkhop_subg(graph, args, new_dataset)
    elif args.sampler == 'shadowkhop':
        new_dataset = gen_shadowkhop_subg(graph, args, new_dataset)
    
    # torch.save(new_dataset, f'../dataset/{args.dataset}-{args.sampler}-5-10-15.pt')
