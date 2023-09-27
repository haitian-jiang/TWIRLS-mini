# python gen_subg.py --dataset=ogbn-products
python gen_subg.py --dataset=ogbn-papers100M --dir=/nvme4n1/offline_subgraph
python gen_subg.py --dataset=mag240m --dir=/nvme4n1/offline_subgraph
python gen_subg.py --dataset=igb-small --dataset_size=small --dir=/nvme4n1/offline_subgraph
python gen_subg.py --dataset=igb-medium --dataset_size=medium --dir=/nvme4n1/offline_subgraph
python gen_subg.py --dataset=igb-large --dataset_size=large --path=/home/ubuntu/dataset/igb_large --dir=/nvme5n1/offline_subgraph
# python gen_subg.py --dataset=igb-full --dataset_size=full --path=/efs/rjliu/dataset/igb_full --dir=/nvme3n1/offline_subgraph --fanout=[10,15]
