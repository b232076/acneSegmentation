import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet_customized import UNetCustom

# Paths to the dataset directories
train_images_dir = '/home/beavpm/inovia/pre_processed_imgs/images/train_sub/imgs'
train_masks_dir  = '/home/beavpm/inovia/pre_processed_imgs/images/train_sub/masks'
val_images_dir   = '/home/beavpm/inovia/pre_processed_imgs/images/val_sub/imgs'
val_masks_dir    = '/home/beavpm/inovia/pre_processed_imgs/images/val_sub/masks'
vis_dir = 'val_preds_vis'
os.makedirs(vis_dir, exist_ok=True)

# Custom Dataset for Acne Segmentation
class AcneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = []
        for fname in sorted(os.listdir(images_dir)):
            mask_name = os.path.splitext(fname)[0] + '.png'
            mask_path = os.path.join(masks_dir, mask_name)
            if os.path.exists(mask_path):
                self.image_files.append(fname)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = os.path.splitext(img_name)[0] + '.png'
        img_path  = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"[WARN] Mask not found, skipping: {mask_path}")
            return self.__getitem__((idx + 1) % len(self))

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask  = self.transform(mask)
        mask = (mask > 0).float()
        return image, mask

# Transforms
transform = transforms.ToTensor()

# Loaders
train_dataset = AcneDataset(train_images_dir, train_masks_dir, transform=transform)
val_dataset   = AcneDataset(val_images_dir, val_masks_dir, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader    = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetCustom().to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0], device=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 20

train_losses = []
val_losses = []
val_dice_scores = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    tp = fp = fn = 0
    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = torch.sigmoid(outputs) > 0.5
            tp += torch.sum((preds == 1) & (masks == 1)).item()
            fp += torch.sum((preds == 1) & (masks == 0)).item()
            fn += torch.sum((preds == 0) & (masks == 1)).item()
            if i < 5:
                img_np  = (images.squeeze(0).cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
                mask_np = (masks.squeeze(0).cpu().numpy().squeeze() * 255).astype(np.uint8)
                pred_np = (preds.squeeze(0).cpu().numpy().squeeze() * 255).astype(np.uint8)
                img_pil  = Image.fromarray(img_np)
                mask_pil = Image.fromarray(mask_np).convert("RGB")
                pred_pil = Image.fromarray(pred_np).convert("RGB")
                merged = Image.new('RGB', (img_pil.width * 3, img_pil.height))
                merged.paste(img_pil, (0, 0))
                merged.paste(mask_pil, (img_pil.width, 0))
                merged.paste(pred_pil, (img_pil.width * 2, 0))
                merged.save(os.path.join(vis_dir, f'epoch{epoch+1}_sample{i+1}.png'))
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        val_dice_scores.append(dice)
    print(f"Época [{epoch+1}/{num_epochs}] Train Loss={avg_train_loss:.4f} Val Loss={avg_val_loss:.4f} Dice={dice:.4f}")

def evaluate(loader, model):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            preds = torch.sigmoid(model(images)) > 0.5
            tp += torch.sum((preds == 1) & (masks == 1)).item()
            fp += torch.sum((preds == 1) & (masks == 0)).item()
            fn += torch.sum((preds == 0) & (masks == 1)).item()
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return dice, iou, prec, rec

# Metrics
train_dice, train_iou, train_prec, train_rec = evaluate(train_loader, model)
val_dice, val_iou, val_prec, val_rec = evaluate(val_loader, model)
print("\n📊 Métricas finais:")
print(f"🔹 Treino -> Dice: {train_dice:.4f} | IoU: {train_iou:.4f} | Precision: {train_prec:.4f} | Recall: {train_rec:.4f}")
print(f"🔸 Validação -> Dice: {val_dice:.4f} | IoU: {val_iou:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f}")

# Graphs
plt.figure(figsize=(8,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Época"); plt.ylabel("Loss"); plt.title("Curva de Loss"); plt.legend()
plt.savefig("loss_curve.png"); plt.close()

plt.figure(figsize=(8,4))
plt.plot(val_dice_scores, label="Val Dice")
plt.xlabel("Época"); plt.ylabel("Dice"); plt.title("Curva de Dice"); plt.legend()
plt.savefig("dice_curve.png"); plt.close()

torch.save(model.state_dict(), "unet_acne.pt")
print("✅ Modelo salvo como unet_acne.pt")
