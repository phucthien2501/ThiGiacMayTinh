import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# Thêm đường dẫn project vào sys.path để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import SGGDataset, collate_fn
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.baseline import BaselineSGG

def train_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    # Đường dẫn
    data_dir = r"SGG\data"
    json_path = os.path.join(data_dir, "annotations.json")
    img_dir = os.path.join(data_dir, "images")
    vocab_path = os.path.join(data_dir, "rel_vocab.json")
    weights_path = os.path.join(data_dir, "rel_weights.json")
    
    # Đọc vocab để lấy số lượng class
    with open(vocab_path, 'r', encoding='utf-8') as f:
        rel_vocab = json.load(f)
    num_classes = len(rel_vocab)
    
    # Load dataset
    full_dataset = SGGDataset(json_path, img_dir, vocab_path, get_train_transforms())
    
    # Chia tập train/val thu nhỏ để huấn luyện thử (Trial Training)
    # Lấy 100 ảnh để train và 20 ảnh để val
    subset_indices = list(range(120))
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    
    train_size = 100
    val_size = 20
    train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])
    
    # Đổi transforms của tập val
    val_dataset.dataset.transforms = get_val_transforms()
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Khởi tạo mô hình
    model = BaselineSGG(num_relations=num_classes, freeze_clip=True).to(device)
    
    # Đọc class weights để gán vào hàm Loss (tránh mất cân bằng dữ liệu)
    with open(weights_path, 'r', encoding='utf-8') as f:
        rel_weights_dict = json.load(f)
    
    weights_tensor = torch.ones(num_classes)
    for idx_str, w in rel_weights_dict.items():
        weights_tensor[int(idx_str)] = w
    weights_tensor = weights_tensor.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    epochs = 2 # Train thử nghiệm 2 epochs
    print("Bắt đầu huấn luyện...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            
            logits, target_tensor = model(images, targets)
            
            if logits is None: # Batch không có relationship hợp lệ
                continue
                
            loss = criterion(logits, target_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target_tensor).sum().item()
            total += target_tensor.size(0)
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        train_acc = correct / total if total > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                logits, target_tensor = model(images, targets)
                if logits is None:
                    continue
                loss = criterion(logits, target_tensor)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == target_tensor).sum().item()
                val_total += target_tensor.size(0)
                
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {total_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
    # Save model
    os.makedirs(r"SGG\models", exist_ok=True)
    torch.save(model.state_dict(), r"SGG\models\baseline_clip.pth")
    print("Đã lưu mô hình tại SGG/models/baseline_clip.pth")

if __name__ == '__main__':
    train_baseline()
