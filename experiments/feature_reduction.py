import argparse
from igb.dataloader import IGB260M
from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np
import os.path as osp
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/efs/rjliu/dataset/igb_dataset", help="path containing the datasets")
parser.add_argument("--dataset_size", type=str, default="tiny", choices=["tiny", "small", "medium", "large", "full"], help="size of the datasets")
parser.add_argument("--num_classes", type=int, default=19, choices=[19, 2983], help="number of classes")
parser.add_argument("--in_memory", type=int, default=1, choices=[0, 1], help="0:read only mmap_mode=r, 1:load into memory")
parser.add_argument("--synthetic", type=int, default=0, choices=[0, 1], help="0:nlp-node embeddings, 1:random")
args = parser.parse_args()

print('Loading Dataset')
dataset = IGB260M(
    root=args.path,
    size=args.dataset_size,
    in_memory=args.in_memory,
    classes=args.num_classes,
    synthetic=args.synthetic,
)
node_features = dataset.paper_feat
print(node_features.shape)
num_nodes = node_features.shape[0]
print('Dataset Loaded')

incre_pca = IncrementalPCA(n_components=128)

batch_size = 10000000
num_batch = (num_nodes + batch_size - 1) // batch_size
for i in tqdm(range(num_batch)):
    start = i * batch_size
    end = num_nodes if i == num_batch - 1 else (i + 1) * batch_size
    partial_feats = node_features[start:end]
    incre_pca.partial_fit(partial_feats)

# Computed mean per feature
mean = incre_pca.mean_
# and stddev
stddev = np.sqrt(incre_pca.var_)

Xtransformed = None
for i in tqdm(range(num_batch)):
    start = i * batch_size
    end = num_nodes if i == num_batch - 1 else (i + 1) * batch_size
    partial_feats = node_features[start:end]
    Xchunk = incre_pca.transform(partial_feats)
    if Xtransformed is None:
        Xtransformed = Xchunk
    else:
        Xtransformed = np.vstack((Xtransformed, Xchunk))

print(Xtransformed.shape)

if args.dataset_size == 'large' or args.dataset_size == 'full':
    path = osp.join(args.path, 'full', 'processed', 'paper', f'node_feat_{128}.npy')
else:
    path = osp.join(args.path, args.dataset_size, 'processed', 'paper', f'node_feat_{128}.npy')
np.save(path, Xtransformed)
