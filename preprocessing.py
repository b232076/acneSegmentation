import os
import cv2 as cv
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import random

class AcneDataset(Dataset):
    def __init__(self, root_dir, image_size=(224, 224), augment=True, save_dir=None, n_augments=3):
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.save_dir = save_dir
        self.n_augments = n_augments

        # List all subdirectories
        self.person_dirs = sorted([
            os.path.join(root_dir, person) for person in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, person))
        ])

        # Collect all image paths and associate them with person labels
        self.image_paths = []
        self.person_labels = []
        for person_dir in self.person_dirs:
            person_name = os.path.basename(person_dir)
            image_files = [
                os.path.join(person_dir, fname) for fname in os.listdir(person_dir)
                if fname.lower().endswith((".jpg", ".jpeg"))
            ]
            self.image_paths.extend(image_files)
            self.person_labels.extend([person_name] * len(image_files))

        # Create save directories for each person
        if self.save_dir:
            for person in set(self.person_labels):
                os.makedirs(os.path.join(self.save_dir, "images", person), exist_ok=True)

        # Define color jitter transformation for augmentation
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.3, saturation=0.3, hue=0.05
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and its associated person label
        image_path = self.image_paths[idx]
        person_name = self.person_labels[idx]
        filename = os.path.splitext(os.path.basename(image_path))[0]

        # Read image using OpenCV, convert to RGB, resize, convert to PIL
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = F.resize(image, self.image_size)

        # Convert to tensor and normalize to [-1, 1]
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(image_tensor, mean=[0.5]*3, std=[0.5]*3)

        # Save original image if save_dir is set
        if self.save_dir:
            save_path = os.path.join(self.save_dir, "images", person_name, f"{filename}.jpg")
            img_np = image_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5
            img_np = (img_np * 255).astype(np.uint8)
            cv.imwrite(save_path, cv.cvtColor(img_np, cv.COLOR_RGB2BGR))

        # Create list of augmented tensors
        augmented_tensors = []
        if self.augment:
            for i in range(self.n_augments):
                aug_img = image

                # Apply random augmentations
                if random.random() < 0.5:
                    aug_img = F.hflip(aug_img)
                if random.random() < 0.2:
                    aug_img = F.vflip(aug_img)
                angle = random.uniform(-30, 30)
                aug_img = F.rotate(aug_img, angle)
                if random.random() < 0.8:
                    aug_img = self.color_jitter(aug_img)
                if random.random() < 0.1:
                    aug_img = F.to_grayscale(aug_img, num_output_channels=3)

                # Convert augmented image to tensor and normalize
                aug_tensor = F.to_tensor(aug_img)
                aug_tensor = F.normalize(aug_tensor, mean=[0.5]*3, std=[0.5]*3)
                augmented_tensors.append(aug_tensor)

                # Save augmented image
                if self.save_dir:
                    aug_np = aug_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5
                    aug_np = (aug_np * 255).astype(np.uint8)
                    aug_path = os.path.join(self.save_dir, "images", person_name, f"{filename}_aug{i}.jpg")
                    cv.imwrite(aug_path, cv.cvtColor(aug_np, cv.COLOR_RGB2BGR))

        return image_tensor, augmented_tensors

    

dataset = AcneDataset(
    root_dir="/home/beavpm/inovia/images",  # Directory with images
    image_size=(224, 224),
    augment=True,
    save_dir="/home/beavpm/inovia/pre_processed_imgs"  # Directory to save pre-processed images
)

# Pre-process the dataset
for image_tensor, augmented_tensors in dataset:
    pass

# Show a few pre-processed images
preprocessed_images_dir = "/home/beavpm/inovia/pre_processed_imgs/images"
image_files = sorted([
    os.path.join(root, fname)
    for root, _, files in os.walk(preprocessed_images_dir)
    for fname in files if fname.lower().endswith((".jpg", ".jpeg"))
])[:4]

fig, axes = plt.subplots(1, len(image_files), figsize=(16, 4))
for ax, img_path in zip(axes, image_files):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis("off")
plt.show()