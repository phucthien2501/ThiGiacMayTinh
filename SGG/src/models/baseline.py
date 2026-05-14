import torch
import torch.nn as nn
from transformers import CLIPVisionModel
import torchvision.transforms.functional as F

class BaselineSGG(nn.Module):
    def __init__(self, num_relations, freeze_clip=True):
        super().__init__()
        # Load pre-trained CLIP
        self.clip = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
                
        # CLIP_ViT_Base_patch32 output dim is 768
        hidden_dim = 768
        
        # Thêm thông tin spatial (tọa độ tương đối của subject và object)
        # spatial_dim = 8 (sx, sy, sw, sh, ox, oy, ow, oh)
        self.spatial_mlp = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_relations)
        )
        
    def crop_union_box(self, image_tensor, sub_box, obj_box):
        """
        Cắt Union Box từ ảnh.
        image_tensor: [C, H, W]
        sub_box, obj_box: [x, y, w, h]
        """
        _, H, W = image_tensor.shape
        
        x1 = min(sub_box[0], obj_box[0])
        y1 = min(sub_box[1], obj_box[1])
        x2 = max(sub_box[0] + sub_box[2], obj_box[0] + obj_box[2])
        y2 = max(sub_box[1] + sub_box[3], obj_box[1] + obj_box[3])
        
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(x2)), min(H, int(y2))
        
        crop = image_tensor[:, y1:y2, x1:x2]
        
        # Fallback nếu crop lỗi
        if crop.shape[1] == 0 or crop.shape[2] == 0:
            crop = image_tensor
            
        crop = F.resize(crop, [224, 224])
        return crop
        
    def forward(self, images, targets):
        """
        images: List[Tensor] (mỗi Tensor [C, H, W])
        targets: List[Dict] chứa 'boxes', 'relationships'
        """
        crops = []
        spatial_feats = []
        valid_targets = [] # Sẽ lưu idx của relationship trong batch
        
        for i, (img, target) in enumerate(zip(images, targets)):
            boxes = target['boxes']
            rels = target['relationships']
            
            _, img_h, img_w = img.shape
            
            for rel in rels:
                if 'relationship_id' not in rel:
                    continue # Bỏ qua nếu ko có ID (out of vocab)
                    
                sub_idx, obj_idx = rel['objects']
                sub_box = boxes[sub_idx]
                obj_box = boxes[obj_idx]
                
                # Crop ảnh
                crop = self.crop_union_box(img, sub_box, obj_box)
                crops.append(crop)
                
                # Spatial features (chuẩn hóa về [0, 1])
                sx, sy, sw, sh = sub_box
                ox, oy, ow, oh = obj_box
                spatial = torch.tensor([
                    sx/img_w, sy/img_h, sw/img_w, sh/img_h,
                    ox/img_w, oy/img_h, ow/img_w, oh/img_h
                ], dtype=torch.float32, device=img.device)
                
                spatial_feats.append(spatial)
                valid_targets.append(rel['relationship_id'])
                
        if len(crops) == 0:
            return None, None
            
        # Gom lại thành batch
        crops_tensor = torch.stack(crops).to(images[0].device)
        spatial_tensor = torch.stack(spatial_feats).to(images[0].device)
        target_tensor = torch.tensor(valid_targets, dtype=torch.long, device=images[0].device)
        
        # Forward pass qua CLIP
        clip_outputs = self.clip(pixel_values=crops_tensor)
        visual_features = clip_outputs.pooler_output # [N, 768]
        
        # Forward qua Spatial MLP
        spatial_features = self.spatial_mlp(spatial_tensor) # [N, 64]
        
        # Concat
        combined = torch.cat([visual_features, spatial_features], dim=1) # [N, 768+64]
        
        # Phân loại
        logits = self.classifier(combined) # [N, num_relations]
        
        return logits, target_tensor
