import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from dataset.vrd_dataset import VRDDataset
from models.gnn_model import SceneGraphModel


def test():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Paths
    json_path = os.path.join(BASE_DIR, "data", "annotations_train.json")
    image_dir = os.path.join(BASE_DIR, "data", "sg_train_images")

    print("📂 Loading dataset...")
    dataset = VRDDataset(json_path, image_dir)
    print(f"✅ Dataset: {len(dataset)} samples, {len(dataset.pred2idx)} relations")

    # Create DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Create model
    model = SceneGraphModel(
        hidden_dim=256,
        num_relations=len(dataset.pred2idx)
    ).to(device)

    print(f"✅ Model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= 1:  # Test only 1 batch
                break
                
            data = data.to(device)
            
            print(f"\n🔄 Batch {i+1}:")
            print(f"   node_imgs: {data.node_imgs.shape}")
            print(f"   edge_index: {data.edge_index.shape}")
            print(f"   edge_attr: {data.edge_attr.shape}")
            
            out = model(data)
            
            print(f"   output: {out.shape}")
            print(f"   target: {data.edge_attr.shape}")
            
            # Loss test
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, data.edge_attr)
            print(f"   loss: {loss.item():.4f}")
            
            break

    print("\n✅ Test passed!")


if __name__ == "__main__":
    test()