import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataset import SGGDataset
from src.data.transforms import get_train_transforms

def visualize_sample(dataset, idx, output_path="visualized_sample.png"):
    """
    Lấy một ảnh từ dataset, vẽ bounding box và Scene Graph.
    """
    image_tensor, target = dataset[idx]
    
    # Chuyển image tensor về numpy để dùng matplotlib
    # Un-normalize (công thức đảo ngược normalize của ImageNet)
    image = image_tensor.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    fig, (ax_img, ax_graph) = plt.subplots(1, 2, figsize=(20, 10))
    
    # 1. Vẽ Ảnh và Bounding Boxes
    ax_img.imshow(image)
    ax_img.set_title(f"Image: {target['filename']}")
    ax_img.axis('off')
    
    boxes = target['boxes'].numpy()
    labels = target['labels']
    for i, box in enumerate(boxes):
        x, y, w, h = box
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax_img.add_patch(rect)
        ax_img.text(x, y-5, f"{i}: {labels[i]}", color='red', fontsize=10, backgroundcolor='white')
        
    # 2. Vẽ Scene Graph bằng NetworkX
    G = nx.DiGraph()
    rels = target['relationships']
    
    # Add nodes
    for i, label in enumerate(labels):
        G.add_node(i, label=f"{i}: {label}")
        
    # Add edges
    edge_labels = {}
    for rel in rels:
        sub_idx, obj_idx = rel['objects']
        rel_name = rel['relationship']
        G.add_edge(sub_idx, obj_idx)
        edge_labels[(sub_idx, obj_idx)] = rel_name
        
    # Điều chỉnh layout để các nhãn xa nhau hơn (tăng k)
    pos = nx.spring_layout(G, k=2.0, iterations=100)
    nx.draw(G, pos, ax=ax_graph, with_labels=True, labels=nx.get_node_attributes(G, 'label'), 
            node_color='lightblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray', 
            arrows=True, arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, ax=ax_graph, edge_labels=edge_labels, font_color='red', font_size=10)
    ax_graph.set_title("Scene Graph")
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Đã lưu ảnh trực quan hoá tại {output_path}")

if __name__ == '__main__':
    json_path = r"c:\Users\Admin\Documents\68CS3\Thị Giác Máy Tính\SGG\data\annotations.json"
    img_dir = r"c:\Users\Admin\Documents\68CS3\Thị Giác Máy Tính\SGG\data\images"
    
    # Tạo dataset với transforms
    transforms = get_train_transforms(image_size=400) # Size to để dễ nhìn
    dataset = SGGDataset(json_path, img_dir, vocab_path=None, transforms=transforms)
    
    # Visualize ảnh đầu tiên
    os.makedirs(r"c:\Users\Admin\Documents\68CS3\Thị Giác Máy Tính\SGG\reports", exist_ok=True)
    visualize_sample(dataset, 0, r"c:\Users\Admin\Documents\68CS3\Thị Giác Máy Tính\SGG\reports\sample_0.png")
