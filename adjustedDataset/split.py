import os
import random
import shutil

# Root path
data_path = "/home/beavpm/inovia/pre_processed_imgs/images"

train_folder = os.path.join(data_path, 'train')
val_folder = os.path.join(data_path, 'val')
test_folder = os.path.join(data_path, 'test')

image_extensions = ['.jpg', '.jpeg']

all_images = []
for root, _, files in os.walk(data_path):
    for file in files:
        if os.path.splitext(file)[-1].lower() in image_extensions:
            if any(part in root for part in ['train', 'val', 'test']):
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, data_path)
            all_images.append(rel_path)

# Separate original and augmented images
original_images = [img for img in all_images if 'aug' not in os.path.basename(img)]
augmented_images = [img for img in all_images if 'aug' in os.path.basename(img)]

# Shuffle with fixed seed
random.seed(42)
random.shuffle(original_images)
random.shuffle(augmented_images)

total_n = len(original_images) + len(augmented_images)
train_n = int(total_n * 0.7)
val_n = int(total_n * 0.15)
test_n = total_n - train_n - val_n

# Test: only original images
test_imgs = original_images[:test_n]
remaining_originals = original_images[test_n:]

# Validation
val_aug_ratio = 0.1  # 10% of val from augmented
val_aug_n = min(int(val_n * val_aug_ratio), len(augmented_images))
val_orig_n = val_n - val_aug_n

val_aug_imgs = augmented_images[:val_aug_n]
remaining_augmented = augmented_images[val_aug_n:]

val_orig_imgs = remaining_originals[:val_orig_n]
remaining_originals = remaining_originals[val_orig_n:]

val_imgs = val_orig_imgs + val_aug_imgs

# Train: all remaining images
train_imgs = remaining_originals + remaining_augmented

def copy_images(image_list, dest_root):
    for rel_path in image_list:
        src = os.path.join(data_path, rel_path)

        parts = os.path.normpath(rel_path).split(os.sep)
        person_id = parts[0]
        filename = parts[1]
        new_name = f"pessoa{person_id}_{filename}"

        dst = os.path.join(dest_root, new_name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

for folder in [train_folder, val_folder, test_folder]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# Copy files
copy_images(train_imgs, train_folder)
copy_images(val_imgs, val_folder)
copy_images(test_imgs, test_folder)

# Summary
print(f" - Total images: {total_n}")
print(f" - Train: {len(train_imgs)} images")
print(f" - Validation: {len(val_imgs)} images (including {len(val_aug_imgs)} augmented)")
print(f" - Test: {len(test_imgs)} images (only original)")
