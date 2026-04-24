import torch
from dataset.vrd_dataset import VRDDataset
from models.gnn_model import SceneGraphModel


def infer():
    dataset = VRDDataset(
        "data/annotations_test.json",
        "data/sg_test_images"
    )

    model = SceneGraphModel(
        hidden_dim=256,
        num_relations=len(dataset.pred2idx)
    )

    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    data = dataset[0]

    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)

    idx2pred = {v: k for k, v in dataset.pred2idx.items()}

    for i, p in enumerate(pred):
        print("Edge:", data.edge_index[:, i].tolist(),
              "Pred:", idx2pred[p.item()])


if __name__ == "__main__":
    infer()