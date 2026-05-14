import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=224):
    """
    Tạo pipeline augmentation cho tập train.
    Cập nhật toạ độ bounding box khi ảnh thay đổi (theo định dạng COCO: [x_min, y_min, width, height]).
    """
    return A.Compose([
        # Có thể tắt Resize nếu muốn giữ ảnh gốc, hoặc resize để model chuẩn (như CLIP 224x224)
        A.Resize(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

def get_val_transforms(image_size=224):
    """
    Pipeline xử lý ảnh cho tập test/val (không augment, chỉ chuẩn hoá).
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))
