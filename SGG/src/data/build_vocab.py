import json
import os
from collections import Counter

def build_vocab(json_path, save_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    rel_counts = Counter()
    obj_counts = Counter()
    
    for item in data:
        for rel in item.get('relationships', []):
            rel_counts[rel['relationship']] += 1
        for obj in item.get('objects', []):
            names = obj.get('names', [])
            if len(names) > 0:
                obj_counts[names[0]] += 1
                
    # Lọc những relations và objects xuất hiện >= 5 lần
    rel_vocab = {rel: idx for idx, (rel, count) in enumerate(rel_counts.most_common()) if count >= 5}
    obj_vocab = {obj: idx for idx, (obj, count) in enumerate(obj_counts.most_common()) if count >= 5}
    
    # Tính toán class weights cho relations (nghịch đảo tần suất)
    total_rels = sum([count for rel, count in rel_counts.items() if count >= 5])
    rel_weights = {}
    for rel, idx in rel_vocab.items():
        weight = total_rels / (rel_counts[rel] * len(rel_vocab))
        rel_weights[idx] = weight
        
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'rel_vocab.json'), 'w') as f:
        json.dump(rel_vocab, f)
    with open(os.path.join(save_dir, 'obj_vocab.json'), 'w') as f:
        json.dump(obj_vocab, f)
    with open(os.path.join(save_dir, 'rel_weights.json'), 'w') as f:
        json.dump(rel_weights, f)
        
    print(f"Đã tạo Vocab: {len(rel_vocab)} relations, {len(obj_vocab)} objects.")
    return rel_vocab, obj_vocab, rel_weights

if __name__ == '__main__':
    build_vocab(r'SGG\data\annotations.json', r'SGG\data')
