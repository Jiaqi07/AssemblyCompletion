import matplotlib
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from data_loader import load_data_from_folder
import torch.optim as optim

import json
import torch
import numpy as np
import yaml
import random

from GCN_Siamese_Attention import GCNModel
from GCN_Siamese_Attention import SiameseNetwork
from GCN_Siamese_Attention import train_model
from find_connections import create_block_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('standard_lego_library.yaml', 'r') as file:
    lego_lib = yaml.safe_load(file)
file.close()

gcn_model = GCNModel(num_node_features=7, filters_1=128, filters_2=64, filters_3=32, bottle_neck_neurons=16, dropout=0.5).to(device)
siamese_model = SiameseNetwork(embedding_dim=32).to(device)
optimizer = optim.Adam(gcn_model.parameters(), lr=0.01)

train_model(gcn_model, siamese_model, optimizer)


def extract_embeddings(data_1, data_2):
    node_embeddings_1 = gcn(data_1)
    h_1 = attention(node_embeddings_1)
    node_embeddings_2 = gcn(data_2)
    h_2 = attention(node_embeddings_2)
    return h_1, h_2


attention = AttentionMechanism(num_classes)

h1, h2 = extract_embeddings(data_cur, data_target)
print(h1, h2)


def similarity_function(h1, h2):
    return cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0))


sim_score = similarity_function(h1, h2)
print(f"Similarity Score: {sim_score.item()}")