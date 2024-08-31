import torch
from torch_geometric.datasets import Planetoid

def load_dataset(name):
    return Planetoid(root=f'./data/{name}', name=name)