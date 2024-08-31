import time
from dataset import load_dataset
from models import GCN, GAT, GraphSAGE, GIN
from torch_geometric.data import DataLoader
from torch_geometric.loader import NeighborSampler


def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
    acc = correct / data.test_mask.sum().item()
    return acc


def main():
    datasets = ['Cora', 'CiteSeer', 'Flickr']
    models = [GCN, GAT, GraphSAGE, GIN]

    for dataset_name in datasets:
        dataset = load_dataset(dataset_name)
        data = dataset[0]
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        subgraph
