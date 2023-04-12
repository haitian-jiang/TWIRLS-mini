import torch
from tqdm import tqdm
from model import GNNModel, APPNP
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import pdb
import argparse


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
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--lmbd', type=float, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=5e-4)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # --- load data --- #
    if args.dataset == 'arxiv':
        dataset = torch.load('./dataset/arxiv.pt')
        output_channels = 40
    elif args.dataset == 'papers100M':
        tr = torch.load('./dataset/papers100M-train.pt')
        va = torch.load('./dataset/papers100M-valid.pt')
        te = torch.load('./dataset/papers100M-test.pt')
        dataset = {'train': tr, 'valid': va, 'test': te}
        output_channels = 172
    input_channels = dataset['train'][0].ndata['feat'].size(1)
    device = torch.device('cuda')


    # --- init model --- #
    if args.model == 'twirls':
        model = GNNModel(input_channels, output_channels, args.hidden, args.K, args.pre_mlp, args.aft_mlp, 'none', True, args.alpha, args.lmbd, dropout=args.dropout)
    # model = SAGE(input_channels, 256, output_channels)
    # model = APPNP(input_channels, [512, 512], output_channels, F.relu, 0.5, 0, 0.05, 10)
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index = -100)
    optimizer = torch.optim.Adam(
        params          = model.parameters() , 
        lr              = args.lr , 
        weight_decay    = args.wd , 
    )

    for e in range(args.epochs):
        # --- train --- #
        tot_loss = 0
        for graph in tqdm(dataset['train']):
            graph = graph.to(device)
            graph.ndata['feature'] = graph.ndata['feat']
            loss = train(model, graph, loss_func, optimizer)
            tot_loss += loss
        print(tot_loss)
        # --- valid ---#
        valid_correct, valid_tot = 0, 0
        for graph in dataset['valid']:
            graph = graph.to(device)
            graph.ndata['feature'] = graph.ndata['feat']
            correct, tot = evaluate(model, graph)
            valid_correct += correct
            valid_tot += tot
        print(f"Valid acc: {valid_correct / valid_tot * 100:.2f}%")
        # --- test --- #
        test_correct, test_tot = 0, 0
        for graph in dataset['test']:
            graph = graph.to(device)
            graph.ndata['feature'] = graph.ndata['feat']
            correct, tot = evaluate(model, graph)
            test_correct += correct
            test_tot += tot
        print(f"Test acc: {test_correct / test_tot * 100:.2f}%")

            


if __name__ == '__main__':
    main()
