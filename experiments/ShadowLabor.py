from numpy.random import default_rng

from dgl import backend as F
from dgl import transforms
from dgl.base import NID
from dgl.random import choice
from dgl.sampling.utils import EidExcluder
from dgl.dataloading.base import Sampler, set_edge_lazy_features, set_node_lazy_features

class ShaDowLabor(Sampler):
    def __init__(
        self,
        fanouts,
        edge_dir="in",
        prob=None,
        importance_sampling=0,
        layer_dependency=False,
        batch_dependency=1,
        prefetch_node_feats=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super().__init__()
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.importance_sampling = importance_sampling
        self.layer_dependency = layer_dependency
        self.prefetch_node_feats=prefetch_node_feats
        self.prefetch_edge_feats=prefetch_edge_feats
        self.output_device=output_device
        self.cnt = F.zeros(2, F.int64, F.cpu())
        self.cnt[0] = -1
        self.cnt[0] = -1
        self.cnt[1] = batch_dependency
        self.random_seed = F.zeros(
            2 if self.cnt[1] > 1 else 1, F.int64, F.cpu()
        )
        self.set_seed(None if batch_dependency > 0 else choice(1e18, 1).item())

    
    def set_seed(self, random_seed=None):
        if random_seed is None:
            self.cnt[0] += 1
            if self.cnt[1] > 0 and self.cnt[0] % self.cnt[1] == 0:
                if self.cnt[0] <= 0 or self.cnt[1] <= 1:
                    if not hasattr(self, "rng"):
                        self.rng = default_rng(choice(1e18, 1).item())
                    self.random_seed[0] = self.rng.integers(1e18)
                    if self.cnt[1] > 1:
                        self.random_seed[1] = self.rng.integers(1e18)
                else:
                    self.random_seed[0] = self.random_seed[1]
                    self.random_seed[1] = self.rng.integers(1e18)
        else:
            self.rng = default_rng(random_seed)
            self.random_seed[0] = self.rng.integers(1e18)
            if self.cnt[1] > 1:
                self.random_seed[1] = self.rng.integers(1e18)
            self.cnt[0] = 0

    def sample(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        for i, fanout in enumerate(reversed(self.fanouts)):
            random_seed_i = F.zerocopy_to_dgl_ndarray(
                self.random_seed + (i if not self.layer_dependency else 0)
            )
            if self.cnt[1] <= 1:
                seed2_contr = 0
            else:
                seed2_contr = ((self.cnt[0] % self.cnt[1]) / self.cnt[1]).item()
            frontier, _ = g.sample_labors(
                seed_nodes,
                fanout,
                edge_dir=self.edge_dir,
                prob=self.prob,
                importance_sampling=self.importance_sampling,
                random_seed=random_seed_i,
                seed2_contribution=seed2_contr,
                output_device=self.output_device,
                exclude_edges=exclude_eids,
            )
            block = transforms.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[NID]
        
        subg = g.subgraph(
            seed_nodes, relabel_nodes=True, output_device=self.output_device
        )
        if exclude_eids is not None:
            subg = EidExcluder(exclude_eids)(subg)

        set_node_lazy_features(subg, self.prefetch_node_feats)
        set_edge_lazy_features(subg, self.prefetch_edge_feats)

        return seed_nodes, output_nodes, subg

    # def sample(self, g, seed_nodes, exclude_eids=None):
    #     import dgl.dataloading.LaborSampler