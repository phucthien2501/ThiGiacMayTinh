import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from dataset.vrd_dataset import VRDDataset
from models.gnn_model import SceneGraphModel


def train():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    json_path = os.path.join(BASE_DIR, "data", "annotations_train.json")
    image_dir = os.path.join(BASE_DIR, "data", "sg_train_images")

    print("📂 JSON:", json_path)
    print("📂 Images:", image_dir)

    dataset = VRDDataset(json_path, image_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SceneGraphModel(
        hidden_dim=256,
        num_relations=len(dataset.pred2idx)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(10):
        total_loss = 0
        step = 0

        for data in loader:
            data = data.to(device)

            if data.edge_attr.size(0) == 0:
                continue

            optimizer.zero_grad()

            out = model(data)
            loss = criterion(out, data.edge_attr)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step += 1

        print(f"Epoch {epoch+1} | Loss: {total_loss/max(step,1):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ Saved model.pth")


if __name__ == "__main__":
    train()