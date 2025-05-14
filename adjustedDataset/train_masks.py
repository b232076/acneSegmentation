import json
import os
import cv2
import numpy as np
from PIL import Image

# Paths
base_dir = "/home/beavpm/inovia/pre_processed_imgs/images/train_sub"
json_path = os.path.join(base_dir, "inovia_acne.json")
img_dir = os.path.join(base_dir, "imgs")
mask_dir = os.path.join(base_dir, "masks")
os.makedirs(mask_dir, exist_ok=True)

# Load VIA annotations
with open(json_path) as f:
    data = json.load(f)

img_annotations = data.get("_via_img_metadata", {})

# Generate mask for each annotated image
for entry in img_annotations.values():
    filename = entry.get("filename")
    regions = entry.get("regions", [])
    img_path = os.path.join(img_dir, filename)

    if not os.path.exists(img_path):
        print(f"[WARN] Image not found: {filename}")
        continue

    with Image.open(img_path) as img:
        width, height = img.size

    mask = np.zeros((height, width), dtype=np.uint8)

    for region in regions:
        shape_attr = region.get("shape_attributes", {})
        if shape_attr.get("name") == "rect":
            x = shape_attr["x"]
            y = shape_attr["y"]
            w = shape_attr["width"]
            h = shape_attr["height"]
            cv2.rectangle(mask, (x, y), (x + w, y + h), color=255, thickness=-1)


    mask_filename = os.path.splitext(filename)[0] + ".png"
    cv2.imwrite(os.path.join(mask_dir, mask_filename), mask)

print("✅ Masks successfully generated at:", mask_dir)
