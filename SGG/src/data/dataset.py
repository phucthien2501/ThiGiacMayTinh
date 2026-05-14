import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SGGDataset(Dataset):
    def __init__(self, json_path, img_dir, vocab_path=None, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        # Load vocab
        self.rel_vocab = {}
        if vocab_path and os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.rel_vocab = json.load(f)
                
        print(f"Đang tải annotations từ {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        self.data = []
        for item in raw_data:
            valid_objects = []
            obj_idx_mapping = {}
            new_idx = 0
            
            img_w = item.get('width', 0)
            img_h = item.get('height', 0)
            
            if img_w <= 0 or img_h <= 0:
                continue
                
            for old_idx, obj in enumerate(item.get('objects', [])):
                bbox = obj.get('bbox', {})
                x, y, w, h = bbox.get('x', 0), bbox.get('y', 0), bbox.get('w', 0), bbox.get('h', 0)
                
                if w > 0 and h > 0:
                    x = max(0, min(x, img_w - 1))
                    y = max(0, min(y, img_h - 1))
                    w = min(w, img_w - x)
                    h = min(h, img_h - y)
                    
                    if w > 0 and h > 0:
                        obj['bbox'] = {'x': x, 'y': y, 'w': w, 'h': h}
                        valid_objects.append(obj)
                        obj_idx_mapping[old_idx] = new_idx
                        new_idx += 1
            
            valid_relations = []
            for rel in item.get('relationships', []):
                objects = rel.get('objects', [])
                rel_name = rel['relationship']
                
                # Ánh xạ sang ID
                if self.rel_vocab and rel_name in self.rel_vocab:
                    rel['relationship_id'] = self.rel_vocab[rel_name]
                    
                if len(objects) == 2:
                    sub_idx, obj_idx = objects
                    if sub_idx in obj_idx_mapping and obj_idx in obj_idx_mapping:
                        rel['objects'] = [obj_idx_mapping[sub_idx], obj_idx_mapping[obj_idx]]
                        valid_relations.append(rel)
                        
            item['objects'] = valid_objects
            item['relationships'] = valid_relations
            
            if len(valid_objects) > 0 and len(valid_relations) > 0:
                self.data.append(item)
                
        print(f"Hoàn tất. Số lượng ảnh hợp lệ dùng để train: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_name = item['filename']
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        
        boxes = []
        labels = []
        for obj in item['objects']:
            bbox = obj['bbox']
            boxes.append([bbox['x'], bbox['y'], bbox['w'], bbox['h']])
            names = obj.get('names', ['unknown'])
            labels.append(names[0] if len(names) > 0 else 'unknown')
            
        if self.transforms and len(boxes) > 0:
            try:
                augmented = self.transforms(image=image_np, bboxes=boxes, class_labels=labels)
                image_tensor = augmented['image']
                boxes = augmented['bboxes']
                labels = augmented['class_labels']
            except Exception as e:
                import torchvision.transforms.functional as F
                image_tensor = F.to_tensor(image)
        else:
            import torchvision.transforms.functional as F
            image_tensor = F.to_tensor(image)
            
        target = {
            'filename': img_name,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.empty((0, 4)),
            'labels': labels,
            'relationships': item['relationships']
        }
        
        return image_tensor, target

# Hàm collate_fn cần thiết vì mỗi batch có số lượng boxes khác nhau
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
