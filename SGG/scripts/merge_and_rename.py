import os
import json
import shutil
from pathlib import Path

def merge_datasets():
    # Đường dẫn thư mục gốc chứa dữ liệu
    base_dir = Path(r"c:\Users\Admin\Documents\68CS3\Thị Giác Máy Tính\data")
    train_img_dir = base_dir / "sg_train_images"
    test_img_dir = base_dir / "sg_test_images"
    train_ann_path = base_dir / "sg_train_annotations.json"
    test_ann_path = base_dir / "sg_test_annotations.json"
    
    # Đường dẫn thư mục đầu ra
    out_dir = Path(r"c:\Users\Admin\Documents\68CS3\Thị Giác Máy Tính\dataset_merged")
    out_img_dir = out_dir / "images"
    out_ann_path = out_dir / "annotations.json"
    
    # Tạo thư mục đầu ra nếu chưa có
    out_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Tải nội dung file annotations
    print("Đang tải file annotations...")
    with open(train_ann_path, 'r', encoding='utf-8') as f:
        train_anns = json.load(f)
    with open(test_ann_path, 'r', encoding='utf-8') as f:
        test_anns = json.load(f)
        
    merged_anns = []
    counter = 1
    
    # Hàm xử lý từng tập dữ liệu
    def process_split(anns, img_dir):
        nonlocal counter
        for ann in anns:
            old_filename = ann.get('filename')
            if not old_filename:
                continue
                
            old_filepath = img_dir / old_filename
            
            # Nếu file ảnh không tồn tại thì bỏ qua (hoặc bạn có thể báo lỗi)
            if not old_filepath.exists():
                print(f"Cảnh báo: Không tìm thấy {old_filepath}. Bỏ qua.")
                continue
                
            new_filename = f"{counter}.jpg"
            new_filepath = out_img_dir / new_filename
            
            # Copy ảnh sang thư mục mới với tên mới
            shutil.copy2(old_filepath, new_filepath)
            
            # Cập nhật tên file (và photo_id nếu có) trong annotations
            ann['filename'] = new_filename
            if 'photo_id' in ann:
                ann['photo_id'] = str(counter)
                
            merged_anns.append(ann)
            counter += 1

    print("Đang xử lý tập train...")
    process_split(train_anns, train_img_dir)
    print("Đang xử lý tập test...")
    process_split(test_anns, test_img_dir)
    
    # Lưu file annotations gộp
    print(f"Đang lưu file annotations gộp tại {out_ann_path}...")
    with open(out_ann_path, 'w', encoding='utf-8') as f:
        json.dump(merged_anns, f, separators=(',', ':'))
        
    print(f"Hoàn tất! Đã gộp thành công {len(merged_anns)} ảnh và nhãn.")

if __name__ == '__main__':
    merge_datasets()
