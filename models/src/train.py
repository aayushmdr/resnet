import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from medical_resnet import get_medical_resnet18 # Ensure this matches your folder logic
from dataset import UterineDataset

# 1. Setup & Paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


weights_path = os.path.join("models", "weights", "resnet_18_23dataset.pth")
metadata_path = "ucec_grading_metadata.csv"

# 2. Initialize Model
model = get_medical_resnet18(weights_path, num_classes=3)
model.to(device)

# 3. Initialize Data (Add this section!)
dataset = UterineDataset(metadata_path)
# Use a small batch size (1-4) for 3D volumes to avoid Out of Memory (OOM) errors
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# 4. DATA GAP STEP 1: Freeze Backbone
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# 5. Training Function
def train_one_epoch(epoch, loader):
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if batch_idx % 5 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch} Average Loss: {total_loss/len(loader):.4f}")

# 6. EXECUTION LOOP (This was missing!)
print("Starting Step 1: Training the Head...")
for epoch in range(1, 6): # Run for 5 epochs
    train_one_epoch(epoch, train_loader)

# 7. DATA GAP STEP 2: Unfreeze Tail (Layer 4)
print("Starting Step 2: Fine-tuning Layer 4...")
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-4}
])

for epoch in range(6, 11): # Run for another 5 epochs
    train_one_epoch(epoch, train_loader)

# Save the trained model
torch.save(model.state_dict(), "models/weights/final_uterine_resnet.pth")