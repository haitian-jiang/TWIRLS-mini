import random
import torch
from tqdm import tqdm
from model import GNNModel, APPNP
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import pdb
import argparse
from torch_geometric.seed import seed_everything


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
    parser.add_argument('--inp_dropout', type=float, default=0.2)
    parser.add_argument('--rec_energy', type=int, default=0)
    parser.add_argument('--activate', type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    seed_everything(args.seed)
    # --- load data --- #
    dataset = torch.load(f'./dataset/{args.dataset}.pt')
    output_channels = dataset['num_classes']
    if args.dataset == 'arxiv':
        output_channels = 40

    # if args.dataset == 'arxiv':
    #     dataset = torch.load('./dataset/arxiv-shadowkhop.pt')
    #     output_channels = 40
    # elif args.dataset == 'papers100M':
    #     # tr = torch.load('./dataset/papers100M-train.pt')
    #     # va = torch.load('./dataset/papers100M-valid.pt')
    #     # te = torch.load('./dataset/papers100M-test.pt')
    #     # dataset = {'train': tr, 'valid': va, 'test': te}
    #     dataset = torch.load('./dataset/papers100M-shadowkhop-5-10-15.pt')
    #     output_channels = 172
    input_channels = dataset['train'][0].ndata['feat'].size(1)
    device = torch.device(f'cuda:{args.device}')


    # --- init model --- #
    if args.model == 'twirls':
        model = GNNModel(input_channels, output_channels, args.hidden, args.K, args.pre_mlp, args.aft_mlp, 'none', args.precond, args.alpha, args.lmbd, dropout=args.dropout, skip=args.skip, inp_dropout=args.inp_dropout, rec_energy=args.rec_energy, activate=args.activate)
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

    def prepare_graph(g):
        g = g.remove_self_loop()
        g = g.add_self_loop()
        return g
    
    dataset['train'] = [prepare_graph(g) for g in dataset['train']]
    dataset['valid'] = [prepare_graph(g) for g in dataset['valid']]
    dataset['test'] = [prepare_graph(g) for g in dataset['test']]

    best_val_acc, best_test_acc, best_epoch = 0, 0, 0

    for e in range(args.epochs):
        # --- train --- #
        tot_loss = 0
        for graph in tqdm(dataset['train']):
            graph = graph.to(device)
            graph.ndata['feature'] = graph.ndata['feat']
            loss = train(model, graph, loss_func, optimizer)
            tot_loss += loss
            random.shuffle(dataset['train'])
        # --- valid ---#
        valid_correct, valid_tot = 0, 0
        for graph in dataset['valid']:
            graph = graph.to(device)
            graph.ndata['feature'] = graph.ndata['feat']
            correct, tot = evaluate(model, graph)
            valid_correct += correct
            valid_tot += tot
        val_acc = valid_correct / valid_tot
        # --- test --- #
        test_correct, test_tot = 0, 0
        for graph in dataset['test']:
            graph = graph.to(device)
            graph.ndata['feature'] = graph.ndata['feat']
            correct, tot = evaluate(model, graph)
            test_correct += correct
            test_tot += tot
        test_acc = test_correct / test_tot
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = e
        if args.log_every > 0 and e % args.log_every == 0:
            print(f"Loss: {tot_loss}\t"
                  f"Valid acc: {val_acc * 100:.2f}%\t"
                  f"Test acc: {test_acc * 100:.2f}%")
    print(f"Best epoch: {best_epoch}")
    print(f"Valid acc: {best_val_acc * 100:.2f}%")
    print(f"Test acc: {best_test_acc * 100:.2f}%")
    if args.rec_energy:
        torch.save(model.unfolding.energy, f'./energy-{args.dataset}.pt')
        breakpoint()


if __name__ == '__main__':
    main()
