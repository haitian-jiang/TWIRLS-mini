import torch
import dgl
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from igb.dataloader import IGB260MDGLDataset
from ogb.lsc import MAG240MDataset
import os


def load_ogb(name, root, to_bidirected=False):
    data = DglNodePropPredDataset(name=name, root=root)
    splitted_idx = data.get_idx_split()
    g, labels = data[0]
    g = g.long()
    if to_bidirected:
        g = dgl.to_bidirected(g, copy_ndata=True)
    feat = g.ndata["feat"]
    labels = labels[:, 0]
    if name == "ogbn-papers100M":
        labels[labels.isnan()] = 404.0
        labels = labels.long()
    n_classes = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g.ndata.clear()
    return g, feat, labels, n_classes, splitted_idx


def load_igb(args, to_bidirected=False):
    data = IGB260MDGLDataset(args)
    g = data[0].long()
    if to_bidirected:
        g = dgl.to_bidirected(g, copy_ndata=True)
    n_classes = args.num_classes
    train_nid = torch.nonzero(g.ndata["train_mask"], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata["test_mask"], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata["val_mask"], as_tuple=True)[0]
    splitted_idx = {"train": train_nid, "test": test_nid, "valid": val_nid}
    feat = g.ndata["feat"]
    labels = g.ndata["label"]
    g.ndata.clear()
    return g, feat, labels, n_classes, splitted_idx


def load_mag240m(root: str, only_graph=True):
    dataset = MAG240MDataset(root=root)
    paper_offset = dataset.num_authors + dataset.num_institutions
    num_nodes = paper_offset + dataset.num_papers
    num_features = dataset.num_paper_features
    (g,), _ = dgl.load_graphs(os.path.join(root, "graph.dgl"))
    g = g.long()
    train_idx = torch.LongTensor(dataset.get_idx_split("train")) + paper_offset
    valid_idx = torch.LongTensor(dataset.get_idx_split("valid")) + paper_offset
    test_idx = torch.LongTensor(dataset.get_idx_split("test-dev")) + paper_offset
    splitted_idx = {"train": train_idx, "test": test_idx, "valid": valid_idx}
    g.ndata.clear()
    feats, label = None, None
    if not only_graph:
        label = torch.from_numpy(dataset.paper_label)
        feats = torch.from_numpy(np.fromfile(os.path.join(root, "full.npy"), dtype="float16").reshape(num_nodes, num_features))
    return g, feats, label, dataset.num_classes, splitted_idx, paper_offset
