import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import dgl
from ogb.nodeproppred import DglNodePropPredDataset

import sys
sys.path.append('../')
from model import UnfoldindAndAttention

class GNNModel(nn.Module):
    def __init__( self , 
        input_d     , 
        prop_step   , 
        precond     = True  ,
        alp         = 0     , 
        lam         = 1     , 
        attention   = False , 
        tau         = 0.2   , 
        T           = -1    , 
        p           = 1     , 
        use_eta     = False ,
        attn_bef    = False , 
        dropout     = 0.0   ,
        attn_dropout= 0.0   , 
    ):
        super().__init__()
        self.input_d        = input_d
        self.prop_step      = prop_step
        self.precond        = precond
        self.attention      = attention
        self.alp            = alp
        self.lam            = lam
        self.tau            = tau
        self.T              = T
        self.p              = p
        self.use_eta        = use_eta
        self.init_att       = attn_bef
        self.dropout        = dropout
        self.attn_dropout   = attn_dropout

        # ----- initialization of some variables -----
        # where to put attention
        self.attn_aft = prop_step // 2 if attention else -1 

        self.unfolding = UnfoldindAndAttention(self.input_d, self.alp, self.lam, self.prop_step, self.attn_aft, 
                self.tau, self.T, self.p, self.use_eta, self.init_att, self.attn_dropout, self.precond)

    def forward(self , g, x=None):
        x = self.unfolding(g , x)
        return x


def load_ogbn_papers100M():
    dataset = DglNodePropPredDataset(name='ogbn-papers100M', root='/home/ubuntu/dataset')
    graph, label = dataset[0]
    g = dgl.to_bidirected(graph, copy_ndata=True)
    g.num_classes = dataset.num_classes
    label = label.view(-1)
    split_idx = dataset.get_idx_split()
    g.split_idx = split_idx
    label = label.view(-1).long()
    g.ndata['label'] = label
    return g

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GNN')
    parser.add_argument('--K', type=int, default=7)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--lmbd', type=float, default=1)
    args = parser.parse_args()
    return args

def index2mask(idx, size: int):
    mask = torch.zeros(size, dtype=torch.bool, device=idx.device)
    mask[idx] = True
    return mask

if __name__ == '__main__':
    args = parse_args()
    print(f"{args.K}\t{args.alpha}\t{args.lmbd}\t", end='')
    sys.stdout.flush()
    # graph = load_ogbn_papers100M()
    # torch.save(graph, '/home/ubuntu/TWIRLS/dataset/100M/bi-graph.pt')
    graph = torch.load('/home/ubuntu/TWIRLS/dataset/100M/bi-graph.pt')
    print(f"Graph loaded")
    sys.stdout.flush()
    input_channels = graph.ndata['feat'].shape[1]
    output_channels = graph.num_classes
    graph.train_mask = index2mask(graph.split_idx['train'], graph.num_nodes())
    graph.valid_mask = index2mask(graph.split_idx['valid'], graph.num_nodes())
    graph.test_mask = index2mask(graph.split_idx['test'], graph.num_nodes())
    graph.mask = graph.train_mask | graph.valid_mask | graph.test_mask

    model = GNNModel(input_channels, args.K, True, args.alpha, args.lmbd)
    model.eval()
    with torch.no_grad():
        embd = model(graph, graph.ndata['feat'])
        embd = embd[graph.mask]
    print(f"embd.shape: {embd.shape}")
    embd = embd.numpy()
    embd.tofile(f'/home/ubuntu/TWIRLS/dataset/100M/{args.K}-{args.alpha}-{args.lmbd}.bin')