import torch
from tqdm import tqdm, trange
import sys
sys.path.append('../')
from model import GNNModel, APPNP
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import pdb
import argparse
from torch_geometric.seed import seed_everything
from ogb.lsc import MAG240MDataset
import numpy as np


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


def train(model, graph, loss_func, optimizer):
    nodes = graph.split_idx
    model.train()
    output = model(graph, graph.ndata['feat'])[nodes]
    labels = graph.ndata['label'][nodes]
    loss = loss_func(output, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, graph):
    model = model.eval()
    nodes = graph.split_idx
    output = model(graph, graph.ndata['feat'])[nodes]
    labels = graph.ndata['label'][nodes]
    correct = (output.argmax(-1) == labels).sum().item()
    total = labels.size(0)
    return correct, total


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GNN')
    parser.add_argument('--dataset', type=str, default='papers100M')
    parser.add_argument('--model', type=str, default='twirls')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--K', type=int, default=7)
    parser.add_argument('--pre_mlp', type=int, default=0)
    parser.add_argument('--aft_mlp', type=int, default=3)
    parser.add_argument('--precond', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--lmbd', type=float, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--activate', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    seed_everything(args.seed)
    # --- load data --- #
    dataset = MAG240MDataset(root='/home/ubuntu/mag')
    output_channels = dataset.num_classes
    paper_offset = dataset.num_authors + dataset.num_institutions
    num_nodes = paper_offset + dataset.num_papers
    num_features = dataset.num_paper_features
    label = dataset.paper_label

    input_channels =  dataset.num_paper_features
    device = torch.device(f'cuda:{args.device}')


    # --- init model --- #
    if args.model == 'twirls':
        model = GNNModel(input_channels, output_channels, args.hidden, args.K, args.pre_mlp, args.aft_mlp, 'none', args.precond, args.alpha, args.lmbd, dropout=args.dropout, skip=args.skip, activate=args.activate)
    # model = SAGE(input_channels, 256, output_channels)
    elif args.model == 'APPNP':
        model = APPNP(input_channels, [args.hidden]*2, output_channels, F.relu, args.dropout, 0, args.alpha, args.K)
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index = -100)
    optimizer = torch.optim.Adam(
        params          = model.parameters() , 
        lr              = args.lr , 
        weight_decay    = args.wd , 
    )

    # def prepare_graph(g):
    #     g = g.remove_self_loop()
    #     g = g.add_self_loop()
    #     return g
    
    feats = np.memmap(
        # "/home/ubuntu/mag/full.npy",
        "/nvme2n1/full.npy",
        mode="r",
        dtype="float16",
        shape=(num_nodes, num_features),
    )
    # feats = np.fromfile('/home/ubuntu/mag/full.npy', dtype=np.float16).reshape(num_nodes, num_features)
    
    # dataset['train'] = [prepare_graph(g) for g in dataset['train']]
    # dataset['valid'] = [prepare_graph(g) for g in dataset['valid']]
    # dataset['test'] = [prepare_graph(g) for g in dataset['test']]

    best_val_acc, best_test_acc, best_epoch = 0, 0, 0

    for e in range(args.epochs):
        # --- train --- #
        tot_loss = 0
        for i in trange(1113):
            graph = torch.load(f'../dataset/MAG240M/shadow-5-15-20/train-{i}.pt')
            # graph = prepare_graph(graph)
            graph.ndata['feat'] = torch.from_numpy(feats[graph.ndata['_ID']]).float()
            graph.ndata['label'] = -1 * torch.zeros_like(graph.ndata['_ID']).long()
            graph.ndata['label'][graph.split_idx] = torch.from_numpy(label[graph.ndata['_ID'][graph.split_idx] - paper_offset]).long()
            graph = graph.to(device)
            loss = train(model, graph, loss_func, optimizer)
            tot_loss += loss
        tot_loss /= 1113
        # --- valid ---#
        valid_correct, valid_tot = 0, 0
        for i in trange(139):
            graph = torch.load(f'../dataset/MAG240M/shadow-5-15-20/valid-{i}.pt')
            # graph = prepare_graph(graph)
            graph.ndata['feat'] = torch.from_numpy(feats[graph.ndata['_ID']]).float()
            graph.ndata['label'] = -1 * torch.zeros_like(graph.ndata['_ID']).long()
            graph.ndata['label'][graph.split_idx] = torch.from_numpy(label[graph.ndata['_ID'][graph.split_idx] - paper_offset]).long()
            graph = graph.to(device)
            correct, tot = evaluate(model, graph)
            valid_correct += correct
            valid_tot += tot
        val_acc = valid_correct / valid_tot
        # --- test --- #
        # test_correct, test_tot = 0, 0
        # for i in trange(89):
        #     graph = torch.load(f'../dataset/MAG240M/shadow-5-15-20/test-{i}.pt')
        #     # graph = prepare_graph(graph)
        #     graph.ndata['feat'] = torch.from_numpy(feats[graph.ndata['_ID']]).float()
        #     graph.ndata['label'] = -1 * torch.zeros_like(graph.ndata['_ID']).long()
        #     graph.ndata['label'][graph.split_idx] = torch.from_numpy(label[graph.ndata['_ID'][graph.split_idx] - paper_offset]).long()
        #     graph = graph.to(device)
        #     correct, tot = evaluate(model, graph)
        #     test_correct += correct
        #     test_tot += tot
        # test_acc = test_correct / test_tot
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # best_test_acc = test_acc
            best_epoch = e
        if args.log_every > 0 and e % args.log_every == 0:
            print("Epoch: {e}\t"
                  f"Loss: {tot_loss}\t"
                  f"Valid acc: {val_acc * 100:.2f}%\t")
                #   f"Test acc: {test_acc * 100:.2f}%")
    print(f"Best epoch: {best_epoch}")
    print(f"Valid acc: {best_val_acc * 100:.2f}%")
    print(f"Test acc: {best_test_acc * 100:.2f}%")


if __name__ == '__main__':
    main()
