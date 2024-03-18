import matplotlib
import torch
import random
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
from data_loader import load_data_from_folder
import networkx as nx
from networkx.algorithms.isomorphism import ISMAGS
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class AttentionModule(torch.nn.Module):
    """
    Attention Module to make a pass on graph.
    """

    def __init__(self, filters_3):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.filters_3 = filters_3
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.filters_3,
                                                             self.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        aggregated_representation = torch.mean(representation, dim=1).view(1, -1)  # Reshape to [1, filters_3]
        return aggregated_representation


class GCNModel(torch.nn.Module):
    def __init__(self, num_node_features, filters_1, filters_2, filters_3, bottle_neck_neurons, dropout):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, filters_1)
        self.conv2 = GCNConv(filters_1, filters_2)
        self.conv3 = GCNConv(filters_2, filters_3)
        self.attention = AttentionModule(filters_3)
        self.fully_connected_first = torch.nn.Linear(filters_3, bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(bottle_neck_neurons, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))

        x = self.attention(x)
        x = F.relu(self.fully_connected_first(x))

        return x


class SiameseNetwork(torch.nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(embedding_dim, 128)  # Note: embedding_dim is already the total concatenated size
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, combined_embeddings):
        x = self.relu(self.fc1(combined_embeddings))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def calculate_mcs_sim(graph1, graph2):
    ismags = ISMAGS(graph1, graph2)
    mcs_subgraphs = list(ismags.largest_common_subgraph())

    if not mcs_subgraphs:
        return 0
    mcs_size = len(mcs_subgraphs[0])

    bigger_graph = max(graph1.number_of_nodes(), graph2.number_of_nodes())
    similarity_score = mcs_size / bigger_graph
    print(similarity_score)
    return torch.tensor(similarity_score, device=device).float()


def to_networkx(data, to_undirected=True):
    G = nx.DiGraph() if not to_undirected else nx.Graph()

    for i in range(data.num_nodes): G.add_node(i)

    edge_index = data.edge_index.cpu().numpy()
    edges = zip(edge_index[0], edge_index[1])
    G.add_edges_from(edges)
    return G


def train_model(gcn_model, siamese_model, optimizer, epochs=100):
    dataset = load_data_from_folder("lego_structures")
    gcn_model.train()
    siamese_model.train()

    for epoch in range(epochs):
        total_loss = 0

        for _ in range(100):
            data_cur = dataset[random.randint(0, len(dataset) - 1)].to(device)
            data_target = dataset[random.randint(0, len(dataset) - 1)].to(device)

            if data_cur.edge_index.numel() == 0 or data_target.edge_index.numel() == 0:
                continue

            optimizer.zero_grad()

            output_cur = gcn_model(data_cur).view(1, -1)
            output_target = gcn_model(data_target).view(1, -1)

            combined_embeddings = torch.cat((output_cur, output_target), dim=1)

            similarity_score = siamese_model(combined_embeddings)
            print(similarity_score)

            nx_graph_cur = to_networkx(data_cur, to_undirected=True)
            nx_graph_target = to_networkx(data_target, to_undirected=True)
            mcs_size = calculate_mcs_sim(nx_graph_cur, nx_graph_target)

            loss = F.mse_loss(similarity_score, mcs_size)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch: {epoch + 1}, Loss: {total_loss / 100}')

        torch.save(gcn_model.state_dict(), f'model/gcn_model_{epochs*100}.pth')
        torch.save(siamese_model.state_dict(), f'model/siamese_model_{epochs*100}.pth')
        print(f'Models saved as {epochs*100}\n')
