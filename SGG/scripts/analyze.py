import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import SGGDataset, collate_fn
from src.data.transforms import get_val_transforms
from src.models.baseline import BaselineSGG

def analyze_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Phân tích trên thiết bị: {device}")
    
    data_dir = r"SGG\data"
    json_path = os.path.join(data_dir, "annotations.json")
    img_dir = os.path.join(data_dir, "images")
    vocab_path = os.path.join(data_dir, "rel_vocab.json")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        rel_vocab = json.load(f)
    num_classes = len(rel_vocab)
    idx_to_rel = {v: k for k, v in rel_vocab.items()}
    
    dataset = SGGDataset(json_path, img_dir, vocab_path, get_val_transforms())
    
    # Lấy ra 20 ảnh cuối cùng để inference nhanh trên CPU
    val_size = 20
    val_dataset = Subset(dataset, range(len(dataset) - val_size, len(dataset)))
    
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    model = BaselineSGG(num_relations=num_classes, freeze_clip=True).to(device)
    model.load_state_dict(torch.load(r"SGG\models\baseline_clip.pth", map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    print("Đang chạy inference trên tập Validation...")
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = [img.to(device) for img in images]
            logits, target_tensor = model(images, targets)
            
            if logits is None:
                continue
                
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target_tensor.cpu().numpy())
            
    # Tính toán Confusion Matrix cho Top N classes
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    print("\n--- BÁO CÁO PHÂN TÍCH LỖI (CLASSIFICATION REPORT) ---")
    report = classification_report(y_true, y_pred, target_names=[idx_to_rel[i] for i in sorted(list(set(y_true).union(set(y_pred))))], zero_division=0)
    print(report)
    
    # Vẽ Confusion Matrix cho 10 class xuất hiện nhiều nhất trong y_true
    unique, counts = np.unique(y_true, return_counts=True)
    top_classes = unique[np.argsort(-counts)][:10] # Top 10
    
    # Lọc ra chỉ những dự đoán thuộc top 10
    mask = np.isin(y_true, top_classes) & np.isin(y_pred, top_classes)
    y_true_top = y_true[mask]
    y_pred_top = y_pred[mask]
    
    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[idx_to_rel[c] for c in top_classes],
                yticklabels=[idx_to_rel[c] for c in top_classes])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Top 10 Classes)')
    
    os.makedirs(r"SGG\reports", exist_ok=True)
    plt.savefig(r"SGG\reports\confusion_matrix.png")
    print("\nĐã lưu Confusion Matrix tại SGG/reports/confusion_matrix.png")

if __name__ == '__main__':
    analyze_model()
