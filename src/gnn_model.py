import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


class GraphSAGEEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1]))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # (num_nodes, out_dim) == h^(L)


class GATEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 128,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
        self.convs.append(GATConv(hidden_dim, out_dim, heads=1, concat=False, dropout=dropout))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def build_gnn_encoder(gnn_type: str, in_dim: int, hidden_dim: int, out_dim: int,
                       num_layers: int, dropout: float):
    if gnn_type == "gat":
        return GATEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout=dropout)
    return GraphSAGEEncoder(in_dim, hidden_dim, out_dim, num_layers, dropout=dropout)


def readout_mean_pool(h, batch):
    return global_mean_pool(h, batch)


class GNNTagClassifier(nn.Module):

    def __init__(self, in_dim: int, num_tags: int, gnn_type: str = "sage",
                 hidden_dim: int = 128, out_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = build_gnn_encoder(gnn_type, in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.head = nn.Linear(out_dim, num_tags)

    def forward(self, x, edge_index, batch):
        h = self.encoder(x, edge_index)          
        g = readout_mean_pool(h, batch)           
        logits = self.head(g)                   
        return logits, g


def bce_tag_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")


class CNNBaseline(nn.Module):
    def __init__(self, num_tags: int, n_mels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, num_tags)

    def forward(self, mel_spec):
        z = self.conv(mel_spec).flatten(1)
        return self.head(z)
