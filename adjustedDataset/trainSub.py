import os
import random
import shutil

train_dir = "/home/beavpm/inovia/pre_processed_imgs/images/train"
subset_dir = "/home/beavpm/inovia/pre_processed_imgs/images/train_sub"

min_n, max_n = 30, 50

if os.path.exists(subset_dir):
    shutil.rmtree(subset_dir)
os.makedirs(subset_dir)

# List of image files in the training folder
image_files = [
    f for f in os.listdir(train_dir)
    if os.path.isfile(os.path.join(train_dir, f)) and f.lower().endswith(('.jpg', '.jpeg'))
]

# Randomly selects between 30 and 50 images
subset_size = random.randint(min_n, max_n)
subset = random.sample(image_files, min(subset_size, len(image_files)))

for filename in subset:
    src = os.path.join(train_dir, filename)
    dst = os.path.join(subset_dir, filename)
    shutil.copy2(src, dst)

print(f"Selected {len(subset)} images for train_sub.")
