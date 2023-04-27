def index2mask(idx, size: int):
    mask = torch.zeros(size, dtype=torch.int, device=idx.device)
    mask[idx] = 1
    return mask

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
train_mask = index2mask(split_idx['train'], graph.num_nodes())
valid_mask = 2*index2mask(split_idx['valid'], graph.num_nodes())
test_mask = 3*index2mask(split_idx['test'], graph.num_nodes())
label_mask = train_mask + valid_mask + test_mask
assert(torch.all(label_mask < 4))
compressed_mask = label_mask[label_mask>0]
compressed_train_mask = compressed_mask == 1
compressed_valid_mask = compressed_mask == 2
compressed_test_mask = compressed_mask == 3
compressed_train_idx = torch.nonzero(compressed_train_mask).view(-1)
compressed_valid_idx = torch.nonzero(compressed_valid_mask).view(-1)
compressed_test_idx = torch.nonzero(compressed_test_mask).view(-1)