import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision
from models.medical_resnet import get_medical_resnet18
from dataset import UterineDataset

# --- SSH / SHARED GPU SETTINGS ---
# Allow other users to use VRAM by not pre-allocating the whole 12GB
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_path = os.path.join("models", "weights", "resnet_18_23dataset.pth")
metadata_path = "ucec_grading_metadata.csv"

# 1. Load Model & Data
model = get_medical_resnet18(weights_path).to(device)
dataset = UterineDataset(metadata_path)
# On 12GB VRAM, batch_size=2 or 4 is safe with 128^3
train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# 2. Optimization with Mixed Precision
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()
scaler = GradScaler() # Scaler for float16 training

def train_one_epoch(epoch, loader):
    model.train()
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Use Autocast for memory efficiency
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if batch_idx % 5 == 0:
            print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}")

# Train loop (example)
for epoch in range(1, 11):
    train_one_epoch(epoch, train_loader)
    torch.save(model.state_dict(), f"models/weights/epoch_{epoch}.pth")