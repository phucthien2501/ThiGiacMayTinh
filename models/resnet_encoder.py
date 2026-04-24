import torch
import torch.nn as nn
import torchvision.models as models


class ResNetEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        # x: [num_nodes, 3, 224, 224]
        batch_size = x.size(0)
        
        # Reshape to process each image
        x = x.view(-1, 3, 224, 224)  # [num_nodes * 1, 3, 224, 224]
        
        x = self.features(x)  # [num_nodes, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [num_nodes, 512]
        
        return self.fc(x)  # [num_nodes, out_dim]