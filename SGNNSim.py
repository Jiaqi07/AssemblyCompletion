import matplotlib
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import json
import torch
import numpy as np
import yaml

matplotlib.use("tkagg")

from find_connections import create_block_graph

f1 = open(
    "C:/Users/Alan/PycharmProjects/AssemblyCompletion/lego_structures/AI/save/it10000-export/task_graph.json")
assembly_list = json.load(f1)

f2 = open(
    "C:/Users/Alan/PycharmProjects/AssemblyCompletion/lego_structures/chair_simple/save/it10000-export/task_graph.json")
target_assembly_list = json.load(f2)

with open('standard_lego_library.yaml', 'r') as file:
    lego_lib = yaml.safe_load(file)
file.close()

cur_node_features = []
for piece_id, piece in assembly_list.items():
    features = [
        piece['x'],
        piece['y'],
        piece['z'],
        lego_lib[piece['brick_id']]['height'],
        lego_lib[piece['brick_id']]['width'],
        piece['ori']
    ]
    cur_node_features.append(features)

target_node_features = []
for piece_id, piece in target_assembly_list.items():
    features = [
        piece['x'],
        piece['y'],
        piece['z'],
        lego_lib[piece['brick_id']]['height'],
        lego_lib[piece['brick_id']]['width'],
        piece['ori']
    ]
    target_node_features.append(features)


def adjacency_list_to_edge_index(adj_list):
    edge_index = []
    node_mapping = {node_id: idx for idx, node_id in enumerate(adj_list.keys())}
    for src, dests in adj_list.items():
        src_idx = node_mapping[src]
        for dest in dests:
            dest_idx = node_mapping[dest]
            edge_index.append([src_idx, dest_idx])
            edge_index.append([dest_idx, src_idx])  # For undirected graph
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


graph_cur = create_block_graph(assembly_list)
graph_target = create_block_graph(target_assembly_list)

edge_index_cur = adjacency_list_to_edge_index(graph_cur)
edge_index_target = adjacency_list_to_edge_index(graph_target)

X = torch.tensor(cur_node_features, dtype=torch.float)
Y = torch.tensor(target_node_features, dtype=torch.float)

data_cur = Data(x=X, edge_index=edge_index_cur)
data_target = Data(x=Y, edge_index=edge_index_target)

# Add node features
data_cur.x = X
data_target.x = Y


# Define the similarity score function
def similarity_function(output1, output2):

    return


class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


model = GNNModel(input_dim=X.size(1), hidden_dim=16, output_dim=10)


class SiameseGNN(torch.nn.Module):
    def __init__(self, gnn_model):
        super(SiameseGNN, self).__init__()
        self.gnn_model = gnn_model

    def forward(self, x1, edge_index1, x2, edge_index2):
        emb1 = self.gnn_model(x1, edge_index1)
        emb2 = self.gnn_model(x2, edge_index2)
        return emb1, emb2


siamese_gnn = SiameseGNN(model)

criterion = torch.nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(siamese_gnn.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()

    emb1 = siamese_gnn.gnn_model(data_cur.x, data_cur.edge_index)
    emb2 = siamese_gnn.gnn_model(data_target.x, data_target.edge_index)
    sim_score = similarity_function(emb1, emb2)

    target = torch.tensor([1], dtype=torch.float)

    loss = criterion(sim_score, target)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}, Similarity Score: {sim_score.item()}")
