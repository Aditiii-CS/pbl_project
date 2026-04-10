import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader

# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])

model = models.efficientnet_b0(weights=None)

# Freeze model
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# Train only classifier
for param in model.classifier.parameters():
    param.requires_grad = True

# Load dataset
dataset = datasets.ImageFolder("dataset", transform=transform)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Load pretrained model
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

# Modify for 2 classes
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(2):
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")

print("Training complete!")

from sklearn.metrics import confusion_matrix

y_true = []
y_pred = []

model.eval()
with torch.no_grad():
    for images, labels in train_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = (np.array(y_true) == np.array(y_pred)).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")