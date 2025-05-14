import json
import os
import cv2
import numpy as np
from PIL import Image

# Paths
base_dir = "/home/beavpm/inovia/pre_processed_imgs/images/val_sub"
json_path = os.path.join(base_dir, "inovia_val.json") 
img_dir = os.path.join(base_dir, "imgs")
mask_dir = os.path.join(base_dir, "masks")
os.makedirs(mask_dir, exist_ok=True)

# Load VIA JSON 
with open(json_path) as f:
    data = json.load(f)

img_annotations = data.get("_via_img_metadata", data)

for key, item in img_annotations.items():
    filename = item.get("filename")
    regions = item.get("regions", [])
    img_path = os.path.join(img_dir, filename)
    if not os.path.exists(img_path):
        print(f"[WARN] Image not found: {filename}")
        continue

    with Image.open(img_path) as img:
        width, height = img.size

    mask = np.zeros((height, width), dtype=np.uint8)

    for region in regions:
        shape = region.get("shape_attributes", {})
        if shape.get("name") == "rect":
            x, y, w, h = shape["x"], shape["y"], shape["width"], shape["height"]
            cv2.rectangle(mask, (x, y), (x + w, y + h), color=255, thickness=-1)
        elif shape.get("name") == "polygon":
            points = np.array(list(zip(shape["all_points_x"], shape["all_points_y"])), dtype=np.int32)
            cv2.fillPoly(mask, [points], color=255)

    mask_filename = os.path.splitext(filename)[0] + ".png"
    cv2.imwrite(os.path.join(mask_dir, mask_filename), mask)

print("✅ Masks saved to:", mask_dir)
