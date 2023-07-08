import random
import argparse
import dgl
import torch
from torch_geometric.seed import seed_everything
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.lsc import MAG240MDataset
from tqdm import trange, tqdm
import numpy as np

def prepare_graph(g):
    g = g.remove_self_loop()
    g = g.add_self_loop()
    return g


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
    parser.add_argument('--fanout', type=str, default='[5,10,15]')
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
    sampler = dgl.dataloading.ShaDowKHopSampler(eval(args.fanout))
    for split, idx in graph.split_idx.items():
        dataloader = dgl.dataloading.DataLoader(graph, idx, sampler, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
        for i, (input_nodes, output_nodes, subgraph) in enumerate(tqdm(dataloader)):
            subgraph = prepare_graph(subgraph)
            idx = torch.arange(len(output_nodes))
            subgraph.split_idx = idx
            subgraph.split = split
            torch.save(subgraph, f'../dataset/MAG240M/shadow-5-15-20/{split}-{i}.pt')
            # new_dataset[split].append(subgraph)
    return new_dataset

if __name__ == '__main__':
    args = parse_arg()
    seed_everything(args.seed)


    dataset = MAG240MDataset(root='/home/ubuntu/mag')
    print("[MAG240M] Loading graph")
    (graph,), _ = dgl.load_graphs('/home/ubuntu/mag/graph.dgl')
    graph = graph.formats(["csc"])
    # print("[MAG240M] Loading features")
    paper_offset = dataset.num_authors + dataset.num_institutions
    num_nodes = paper_offset + dataset.num_papers
    num_features = dataset.num_paper_features


    train_idx = torch.LongTensor(dataset.get_idx_split("train")) + paper_offset
    valid_idx = torch.LongTensor(dataset.get_idx_split("valid")) + paper_offset
    test_idx = torch.LongTensor(dataset.get_idx_split("test-dev")) + paper_offset
    graph.split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    # feats = np.memmap(
    #     args.full_feature_path,
    #     mode="r",
    #     dtype="float16",
    #     shape=(num_nodes, num_features),
    # )

    args.num_classes = dataset.num_classes
    args.num_features = dataset.num_paper_features

    print("Dataset ready")

    new_dataset = {split: [] for split in ['train', 'valid', 'test']}
    new_dataset['num_classes'] = args.num_classes
    new_dataset['num_features'] = args.num_features

    if args.sampler == 'fullkhop':
        new_dataset = gen_fullkhop_subg(graph, args, new_dataset)
    elif args.sampler == 'shadowkhop':
        new_dataset = gen_shadowkhop_subg(graph, args, new_dataset)
    
    # torch.save(new_dataset, f'../dataset/{args.dataset}-{args.sampler}-{args.fanout}.pt')
