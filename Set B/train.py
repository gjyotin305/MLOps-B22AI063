import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
from tqdm.auto import tqdm

# ==========================
# Configuration
# ==========================
DATA_DIR = "data/train/"
BATCH_SIZE = 256
NUM_CLASSES = 10
EPOCHS = 1
LR = 1e-3
# DEVICE="mps"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset & DataLoader
# ==========================
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Classes:", dataset.classes)

# ==========================
# Load ResNet-18
# ==========================
model = models.resnet18(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

model = model.to(DEVICE)

# ==========================
# Loss & Optimizer
# ==========================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def pbar_log(pbar, **metrics):
    """Log metrics on tqdm progress bar."""
    formatted = {}
    for k, v in metrics.items():
        if isinstance(v, float):
            formatted[k] = f"{v:.4f}"
        else:
            formatted[k] = v
    pbar.set_postfix(formatted)

# ==========================
# Training Loop
# ==========================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}",
        unit="batch"
    )

    for batch_idx, (images, labels) in enumerate(pbar, start=1):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        avg_loss = running_loss / batch_idx
        avg_acc = 100 * correct / total
        pbar_log(pbar, loss=avg_loss, acc=avg_acc)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total

    pbar.write(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"Accuracy: {epoch_acc:.2f}%"
    )

print("Training Complete!")

# ==========================
# Save Model
# ==========================
torch.save(model.state_dict(), "setB.pth")
print("Model saved!")
