import os
import random
import shutil

val_dir = "/home/beavpm/inovia/pre_processed_imgs/images/val"
subset_dir = "/home/beavpm/inovia/pre_processed_imgs/images/val_sub/imgs"
os.makedirs(subset_dir, exist_ok=True)

image_extensions = (".jpg", ".jpeg", ".png")
images = [f for f in os.listdir(val_dir) if f.lower().endswith(image_extensions)]

num_images = min(50, len(images))
random.seed(42)
selected = random.sample(images, num_images)

for img_name in selected:
    src = os.path.join(val_dir, img_name)
    dst = os.path.join(subset_dir, img_name)
    shutil.copy2(src, dst)

print(f"✅ {len(selected)} images copied to: {subset_dir}")
