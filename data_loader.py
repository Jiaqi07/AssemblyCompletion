import os
import torch
import json
import yaml
from torch_geometric.data import Data

from find_connections import create_block_graph

with open('standard_lego_library.yaml', 'r') as file:
    lego_lib = yaml.safe_load(file)
file.close()


def load_data_from_folder(root_folder):
    data_sets = []

    def load_from_path(path):
        save_export_path = os.path.join(path, 'save', 'it10000-export', 'task_graph.json')

        if os.path.exists(save_export_path):
            with open(save_export_path, 'r') as file:
                file_data = json.load(file)

                data = convert_graph_to_data_object(file_data)
                if data:
                    data_sets.append(data)
        else:
            for subdir in os.listdir(path):
                subdir_path = os.path.join(path, subdir)
                if os.path.isdir(subdir_path):
                    load_from_path(subdir_path)

    def adjacency_list_to_edge_index(adj_list):
        edge_index = []
        node_mapping = {node_id: idx for idx, node_id in enumerate(adj_list.keys())}
        for src, dests in adj_list.items():
            src_idx = node_mapping[src]
            for dest in dests:
                dest_idx = node_mapping[dest]
                edge_index.append([src_idx, dest_idx])
        return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    def extract_node_features(assembly_list):
        cur_node_features = []
        for piece_id, piece in assembly_list.items():
            features = [
                piece['x'],
                piece['y'],
                piece['z'],
                lego_lib[piece['brick_id']]['height'],
                lego_lib[piece['brick_id']]['width'],
                piece['brick_id'],
                piece['ori']
            ]
            cur_node_features.append(features)
        return torch.tensor(cur_node_features, dtype=torch.float)

    def convert_graph_to_data_object(file_data):
        edge_index = adjacency_list_to_edge_index(create_block_graph(file_data))
        node_features = extract_node_features(file_data)  # Implement this based on your data structure
        # print(edge_index)
        # print(node_features)
        return Data(x=node_features, edge_index=edge_index)

    load_from_path(root_folder)
    return data_sets


if __name__ == '__main__':
    root_folder = "lego_structures"
    dataset = load_data_from_folder(root_folder)
    print(dataset)
