import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Configuration
DATA_DIR = 'PPE_reformatted'
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 2
NUM_CLASSES = len(os.listdir(os.path.join(DATA_DIR, 'train')))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Datasets and Dataloaders
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
NUM_CLASSES = len(train_dataset.classes)
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Datasets and Dataloaders
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
NUM_CLASSES = len(train_dataset.classes)
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_CLASSES)
model.to(DEVICE)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# For plotting
val_losses = []
train_losses = []
val_accuracies = []
train_accuracies = []

from tqdm import tqdm
import time

for epoch in range(EPOCHS):
    start_time = time.time()
    
    model.train()
    running_loss = 0.0
    correct = 0
    
    # tqdm progress bar for batches
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch")
    
    for images, labels in train_loader_tqdm:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

        # Optional: Update tqdm description with loss
        train_loader_tqdm.set_postfix(loss=loss.item())

    train_acc = correct / len(train_dataset)
    epoch_duration = time.time() - start_time
    train_losses.append(running_loss)
    train_accuracies.append(train_acc)

    print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Duration: {epoch_duration:.2f} sec")

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(1)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / len(val_dataset)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, EPOCHS+1), val_losses, marker='o', label='Train Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_accuracies, marker='o', label='Train Acc')
plt.plot(range(1, EPOCHS+1), val_accuracies, marker='o', label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Validation Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

# Print classification report
report = classification_report(all_labels, all_preds, target_names=train_dataset.classes, digits=4)
print("Classification Report:\n")
print(report)





from PIL import Image
import random

# Load and transform a random image from the test set
test_dir = os.path.join(DATA_DIR, 'test')
test_classes = os.listdir(test_dir)
selected_class = random.choice(test_classes)
test_class_dir = os.path.join(test_dir, selected_class)
test_images = os.listdir(test_class_dir)
selected_image = random.choice(test_images)

image_path = os.path.join(test_class_dir, selected_image)
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # Shape: [1, 3, 224, 224]

# Display the image
plt.imshow(image)
plt.title(f"Actual: {selected_class}")
plt.axis('off')
plt.show()

# Inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred_index = probs.argmax(dim=1).item()
    pred_class = train_dataset.classes[pred_index]
    confidence = probs[0, pred_index].item()

print(f"Predicted Class: {pred_class} (Confidence: {confidence*100:.2f}%)")
