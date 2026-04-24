import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from models.resnet_encoder import ResNetEncoder


class SceneGraphModel(nn.Module):
    def __init__(self, hidden_dim, num_relations):
        super().__init__()

        self.encoder = ResNetEncoder(hidden_dim)

        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_relations)
        )

    def forward(self, data):
        # Handle batched node_imgs from DataLoader
        node_imgs = data.node_imgs
        
        # If node_imgs is a list (batch from DataLoader), concatenate
        if isinstance(node_imgs, list):
            node_imgs = torch.cat(node_imgs, dim=0)
        
        x = self.encoder(node_imgs)
        edge_index = data.edge_index

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        src = x[edge_index[0]]
        dst = x[edge_index[1]]

        edge_feat = torch.cat([src, dst], dim=1)
        return self.edge_mlp(edge_feat)