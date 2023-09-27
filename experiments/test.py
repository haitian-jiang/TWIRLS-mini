import dgl
import os
import psutil


# --- load data --- #
process = psutil.Process(os.getpid())
mem = process.memory_info().rss / (1024 * 1024 * 1024)

(g,), _ = dgl.load_graphs(os.path.join("/home/ubuntu/mag", "graph.dgl"))
# g = g.long()
g.ndata.clear()

print(g)
mem1 = process.memory_info().rss / (1024 * 1024 * 1024)
print("Graph total memory:", mem1 - mem, "GB")