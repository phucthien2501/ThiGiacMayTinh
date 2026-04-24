import json

ANNOTATION = "vrd_project/data/annotations_train.json"

with open(ANNOTATION, encoding="utf-8") as f:
    data = json.load(f)

# In thử 5 ảnh đầu
for i, (img_name, relations) in enumerate(data.items()):
    print(img_name)

    if i == 5:
        break