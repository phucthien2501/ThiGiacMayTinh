import json
import numpy as np
from collections import Counter
import torch
from torch.utils.data import WeightedRandomSampler

def get_balanced_sampler(json_path):
    """
    Đọc annotations, đếm tần suất các relationship,
    và trả về WeightedRandomSampler của PyTorch.
    """
    print(f"Đang đếm tần suất relations từ {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    rel_counts = Counter()
    image_weights = []
    
    # Đếm tổng tần suất của từng loại relationship trong toàn bộ dataset
    for item in data:
        for rel in item.get('relationships', []):
            rel_counts[rel['relationship']] += 1
            
    print(f"Top 5 relations phổ biến: {rel_counts.most_common(5)}")
    
    # Tính trọng số cho từng quan hệ (Nghịch đảo của tần suất để phạt các lớp đa số)
    total_rels = sum(rel_counts.values())
    rel_weights = {k: total_rels / v for k, v in rel_counts.items()}
    
    # Tính trọng số của mỗi bức ảnh dựa trên các relationship nó chứa
    for item in data:
        weight = 0.0
        rels = item.get('relationships', [])
        if len(rels) == 0:
            weight = 1.0 # Giá trị mặc định nếu không có relation
        else:
            for rel in rels:
                weight += rel_weights[rel['relationship']]
            weight /= len(rels) # Lấy trung bình
        image_weights.append(weight)
        
    weights_tensor = torch.DoubleTensor(image_weights)
    sampler = WeightedRandomSampler(weights=weights_tensor, num_samples=len(weights_tensor), replacement=True)
    
    print("Đã tạo xong WeightedRandomSampler để cân bằng dữ liệu.")
    return sampler
