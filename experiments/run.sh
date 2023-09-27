# python gen_subg.py --dataset=ogbn-products
# python gen_subg.py --dataset=ogbn-papers100M
# python gen_subg.py --dataset=mag240m
# python gen_subg.py --dataset=igb-small --dataset_size=small
# python gen_subg.py --dataset=igb-medium --dataset_size=medium
# python gen_subg.py --dataset=igb-large --dataset_size=large --path=/home/ubuntu/dataset/igb_large --dir=/nvme2n1/offline_subgraph --fanout=[10,15]
# python gen_subg.py --dataset=igb-full --dataset_size=full --path=/efs/rjliu/dataset/igb_full --dir=/nvme3n1/offline_subgraph --fanout=[10,15]
# python gen_subg.py --dataset=igb-large --dataset_size=large --path=/home/ubuntu/dataset/igb_large --dir=/nvme6n1/offline_subgraph --fanout='[10,15]'
# python gen_subg.py --dataset=igb-full --dataset_size=full --path=/efs/rjliu/dataset/igb_full --dir=/nvme7n1/offline_subgraph --fanout='[10,15]'


# python feature_reduction.py --dataset_size=small
# python feature_reduction.py --dataset_size=medium
# python feature_reduction.py --dataset_size=large --path=/efs/rjliu/dataset/igb_large
# python feature_reduction.py --dataset_size=full --path=/efs/rjliu/dataset/igb_full


# python offline_shadow.py --dataset=ogbn-products --device=0 --epochs=100 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=0 --precond=1 --alpha=1 --lmbd=1
# python online.py --dataset=ogbn-products --device=0 --epochs=6 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=0 --precond=1 --alpha=1 --lmbd=1

# python offline_shadow.py --dataset=ogbn-papers100M --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4
# python online.py --dataset=ogbn-papers100M --epochs=6 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4

# python offline_shadow.py --dataset=mag240m --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4
# python online.py --dataset=mag240m --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4

## python offline.py --dataset=igb-small --dataset_size=small --device=0 --epochs=6 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=0 --precond=1 --alpha=1 --lmbd=1
## python offline.py --dataset=igb-small --dataset_size=small --device=2 --epochs=20 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=0 --aft_mlp=3 --skip=0 --precond=1 --alpha=1 --lmbd=1 > 'log/igb_small/512-8-[0,3]-1.txt'
## python offline.py --dataset=igb-small --dataset_size=small --device=3 --epochs=20 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=1 --alpha=1 --lmbd=1 > 'log/igb_small/512-8-[3+,3+]-1.txt'
# python offline_shadow.py --dataset=igb-small --dataset_size=small --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1
# python online.py --dataset=igb-small --dataset_size=small --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1

# python offline_shadow.py --dataset=igb-medium --dataset_size=medium --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1
# python online.py --dataset=igb-medium --dataset_size=medium --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1

# python offline_shadow.py --dataset=igb-large --dataset_size=large --path=/home/ubuntu/dataset/igb_large --fanout='[10,15]' --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1 --dir=/nvme2n1/offline_subgraph
# python offline_ns.py --dataset=igb-large --dataset_size=large --path=/home/ubuntu/dataset/igb_large --fanout='[10,15]' --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1 --dir=/nvme6n1/offline_subgraph
# python online.py --dataset=igb-large --dataset_size=large --path=/efs/rjliu/dataset/igb_large --fanout='[10,15]' --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1

# python offline_shadow.py --dataset=igb-full --dataset_size=full --path=/efs/rjliu/dataset/igb_full --fanout='[10,15]' --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1 --dir=/nvme3n1/offline_subgraph
# python offline_ns.py --dataset=igb-full --dataset_size=full --path=/efs/rjliu/dataset/igb_full --fanout='[10,15]' --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1 --dir=/nvme7n1/offline_subgraph
# python online.py --dataset=igb-full --dataset_size=full --path=/efs/rjliu/dataset/igb_full --fanout='[10,15]' --device=0 --epochs=6 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1



# python offline.py --dataset=igb-medium --dataset_size=medium --device=0 --epochs=10 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1 > 'log/igb_medium/512-8-[3+,3+]-0-0.2-relu.txt'
# python offline.py --dataset=igb-large --dataset_size=large --path=/efs/rjliu/dataset/igb_large --fanout='[10,15]' --device=0 --epochs=10 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1 --dir=/nvme2n1/offline_subgraph > 'log/igb_large/512-8-[3+,3+]-0-0.2-relu.txt'
# python offline.py --dataset=igb-full --dataset_size=full --path=/efs/rjliu/dataset/igb_full --fanout='[10,15]' --device=0 --epochs=10 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1 --dir=/nvme3n1/offline_subgraph > 'log/igb_full/512-8-[3+,3+]-0-0.2-relu.txt'


# python baseline.py --dataset=ogbn-products --epochs=3 --model=SAGE
# python baseline.py --dataset=ogbn-papers100M --hidden=512 --epochs=3 --model=SAGE
# python baseline.py --dataset=mag240m --hidden=1024 --epochs=3 --model=SAGE
# python baseline.py --dataset=igb-small --dataset_size=small --epochs=3 --model=SAGE
# python baseline.py --dataset=igb-medium --dataset_size=medium --epochs=3 --model=SAGE
# python baseline.py --dataset=igb-large --dataset_size=large --path=/home/ubuntu/dataset/igb_large --epochs=3 --model=SAGE --fanout='[10,15]' --hidden=512
# python baseline.py --dataset=igb-full --dataset_size=full --path=/efs/rjliu/dataset/igb_full --epochs=3 --model=SAGE --fanout='[10,15]' --hidden=512

# python baseline.py --dataset=ogbn-products --epochs=3 --model=GAT
# python baseline.py --dataset=ogbn-papers100M --hidden=512 --epochs=3 --model=GAT
# python baseline.py --dataset=mag240m --hidden=512 --epochs=3 --model=GAT
# python baseline.py --dataset=igb-small --dataset_size=small --epochs=3 --model=GAT
# python baseline.py --dataset=igb-medium --dataset_size=medium --epochs=3 --model=GAT
# python baseline.py --dataset=igb-large --dataset_size=large --path=/home/ubuntu/dataset/igb_large --epochs=3 --model=GAT --fanout='[10,15]' --hidden=512
# python baseline.py --dataset=igb-full --dataset_size=full --path=/efs/rjliu/dataset/igb_full --epochs=3 --model=GAT --fanout='[10,15]' --hidden=512




# python offline_ns.py --dataset=ogbn-products --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=0 --precond=1 --alpha=1 --lmbd=1
# python offline_ns.py --dataset=ogbn-papers100M --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1
# python offline_ns.py --dataset=mag240m --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4
# python offline_ns.py --dataset=igb-small --dataset_size=small --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1
# python offline_ns.py --dataset=igb-medium --dataset_size=medium --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1
# python offline_ns.py --dataset=igb-large --dataset_size=large --path=/home/ubuntu/dataset/igb_large --device=0 --epochs=3 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1 --dir=/nvme5n1/offline_subgraph


# python offline_shadow.py --dataset=igb-large --dataset_size=large --path=/home/ubuntu/dataset/igb_large --fanout='[10,15]' --epochs=10 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1 --dir=/nvme2n1/offline_subgraph --device=0 > 'log/igb_large/512-8-[3+,3+]-0-0.2-relu-0.txt'

python offline_shadow.py --dataset=igb-full --dataset_size=full --path=/efs/rjliu/dataset/igb_full --fanout='[10,15]' --epochs=10 --hidden=512 --K=8 --dropout=0.2 --lr=1e-3 --wd=0 --pre_mlp=3 --aft_mlp=3 --skip=1 --precond=0 --alpha=0.2 --lmbd=4 --activate=1 --dir=/nvme3n1/offline_subgraph --device=0 > 'log/igb_full/512-8-[3+,3+]-0-0.2-relu-0.txt'
